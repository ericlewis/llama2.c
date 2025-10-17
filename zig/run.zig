const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

pub const Config = struct {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,

    pub fn headSize(self: Config) usize {
        return self.dim / self.n_heads;
    }

    pub fn kvDim(self: Config) usize {
        return (self.dim * self.n_kv_heads) / self.n_heads;
    }
};

pub const TransformerWeights = struct {
    token_embedding_table: []const f32,
    rms_att_weight: []const f32,
    rms_ffn_weight: []const f32,
    wq: []const f32,
    wk: []const f32,
    wv: []const f32,
    wo: []const f32,
    w1: []const f32,
    w2: []const f32,
    w3: []const f32,
    rms_final_weight: []const f32,
    wcls: []const f32,
};

const RopeFreqCache = struct {
    inv_freq: []f32,

    pub fn init(allocator: Allocator, head_size: usize) !RopeFreqCache {
        if (head_size < 2) {
            return RopeFreqCache{ .inv_freq = &[_]f32{} };
        }
        const count = head_size / 2;
        var buf = try allocator.alloc(f32, count);
        const inv_head_size = 2.0 / @as(f64, @floatFromInt(head_size));
        var i: usize = 0;
        while (i < count) : (i += 1) {
            buf[i] = @as(f32, @floatCast(math.pow(f64, 10000.0, -@as(f64, @floatFromInt(i)) * inv_head_size)));
        }
        return RopeFreqCache{ .inv_freq = buf };
    }

    pub fn deinit(self: *RopeFreqCache, allocator: Allocator) void {
        if (self.inv_freq.len > 0) {
            allocator.free(self.inv_freq);
        }
        self.inv_freq = &[_]f32{};
    }
};

pub const RopeCosSinCache = struct {
    cos: []f32,
    sin: []f32,
    seq_len: usize,
    head_size: usize,

    pub fn init(allocator: Allocator, config: Config, freq: RopeFreqCache) !RopeCosSinCache {
        const head_size = config.headSize();
        if (freq.inv_freq.len == 0) {
            return RopeCosSinCache{
                .cos = &[_]f32{},
                .sin = &[_]f32{},
                .seq_len = config.seq_len,
                .head_size = head_size,
            };
        }
        const cache_len = config.seq_len * freq.inv_freq.len;
        var cos_table = try allocator.alloc(f32, cache_len);
        var sin_table = try allocator.alloc(f32, cache_len);
        var t: usize = 0;
        while (t < config.seq_len) : (t += 1) {
            const base_index = t * freq.inv_freq.len;
            const theta_t = @as(f32, @floatFromInt(t));
            var i: usize = 0;
            while (i < freq.inv_freq.len) : (i += 1) {
                const angle = theta_t * freq.inv_freq[i];
                var s: f64 = 0;
                var c: f64 = 0;
                math.sincos(@as(f64, angle), &s, &c);
                cos_table[base_index + i] = @as(f32, @floatCast(c));
                sin_table[base_index + i] = @as(f32, @floatCast(s));
            }
        }
        return RopeCosSinCache{
            .cos = cos_table,
            .sin = sin_table,
            .seq_len = config.seq_len,
            .head_size = head_size,
        };
    }

    pub fn deinit(self: *RopeCosSinCache, allocator: Allocator) void {
        if (self.cos.len > 0) allocator.free(self.cos);
        if (self.sin.len > 0) allocator.free(self.sin);
        self.* = RopeCosSinCache{ .cos = &[_]f32{}, .sin = &[_]f32{}, .seq_len = 0, .head_size = 0 };
    }

    pub fn lookup(self: RopeCosSinCache, pos: usize) RopePosView {
        if (self.cos.len == 0) {
            return RopePosView.empty();
        }
        const stride = self.cos.len / self.seq_len;
        const offset = pos * stride;
        return RopePosView{
            .cos = self.cos[offset .. offset + stride],
            .sin = self.sin[offset .. offset + stride],
        };
    }
};

pub const RopePosView = struct {
    cos: []const f32,
    sin: []const f32,

    pub fn empty() RopePosView {
        return .{ .cos = &[_]f32{}, .sin = &[_]f32{} };
    }
};

pub const RunState = struct {
    x: []f32,
    xb: []f32,
    xb2: []f32,
    hb: []f32,
    hb2: []f32,
    q: []f32,
    att: []f32,
    logits: []f32,
    key_cache: []f32,
    value_cache: []f32,
    rope_freq: RopeFreqCache,
    rope_cache: RopeCosSinCache,

    pub fn init(allocator: Allocator, config: Config) !RunState {
        const kv_dim = config.kvDim();
        const head_size = config.headSize();
        var state = RunState{
            .x = try allocator.alloc(f32, config.dim),
            .xb = try allocator.alloc(f32, config.dim),
            .xb2 = try allocator.alloc(f32, config.dim),
            .hb = try allocator.alloc(f32, config.hidden_dim),
            .hb2 = try allocator.alloc(f32, config.hidden_dim),
            .q = try allocator.alloc(f32, config.dim),
            .att = try allocator.alloc(f32, config.n_heads * config.seq_len),
            .logits = try allocator.alloc(f32, config.vocab_size),
            .key_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * kv_dim),
            .value_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * kv_dim),
            .rope_freq = try RopeFreqCache.init(allocator, head_size),
            .rope_cache = RopeCosSinCache{ .cos = &[_]f32{}, .sin = &[_]f32{}, .seq_len = 0, .head_size = head_size },
        };
        try state.materializeRopeCache(allocator, config);
        return state;
    }

    pub fn materializeRopeCache(self: *RunState, allocator: Allocator, config: Config) !void {
        if (self.rope_cache.cos.len != 0 or self.rope_cache.sin.len != 0) {
            self.rope_cache.deinit(allocator);
        }
        self.rope_cache = try RopeCosSinCache.init(allocator, config, self.rope_freq);
    }

    pub fn deinit(self: *RunState, allocator: Allocator) void {
        allocator.free(self.x);
        allocator.free(self.xb);
        allocator.free(self.xb2);
        allocator.free(self.hb);
        allocator.free(self.hb2);
        allocator.free(self.q);
        allocator.free(self.att);
        allocator.free(self.logits);
        allocator.free(self.key_cache);
        allocator.free(self.value_cache);
        self.rope_cache.deinit(allocator);
        self.rope_freq.deinit(allocator);
    }
};

fn ropeRotate(qkv: []f32, head_size: usize, pos: usize, cache: RopeCosSinCache) void {
    if (cache.cos.len == 0) return;
    const freq = cache.lookup(pos);
    var h: usize = 0;
    while (h < qkv.len) : (h += head_size) {
        var i: usize = 0;
        var pair_index: usize = 0;
        while (i < head_size) : (i += 2) {
            const cos_theta = freq.cos[pair_index];
            const sin_theta = freq.sin[pair_index];
            const x0 = qkv[h + i];
            const x1 = qkv[h + i + 1];
            qkv[h + i] = x0 * cos_theta - x1 * sin_theta;
            qkv[h + i + 1] = x0 * sin_theta + x1 * cos_theta;
            pair_index += 1;
        }
    }
}

fn matmul_simd(out: []f32, inp: []const f32, weight: []const f32, rows: usize, cols: usize) void {
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var acc0: f32 = 0.0;
        var acc1: f32 = 0.0;
        var acc2: f32 = 0.0;
        var acc3: f32 = 0.0;
        var c: usize = 0;
        while (c + 4 <= cols) : (c += 4) {
            const base = r * cols + c;
            acc0 += inp[c] * weight[base];
            acc1 += inp[c + 1] * weight[base + 1];
            acc2 += inp[c + 2] * weight[base + 2];
            acc3 += inp[c + 3] * weight[base + 3];
        }
        var acc = (acc0 + acc1) + (acc2 + acc3);
        while (c < cols) : (c += 1) {
            acc += inp[c] * weight[r * cols + c];
        }
        out[r] = acc;
    }
}

fn rmsNorm(out: []f32, inp: []const f32, weight: []const f32) void {
    var ss: f32 = 0.0;
    for (inp) |v| {
        ss += v * v;
    }
    const mean = ss / @as(f32, @floatFromInt(inp.len));
    const inv = math.rsqrt(mean + 1e-5);
    var i: usize = 0;
    while (i < inp.len) : (i += 1) {
        out[i] = weight[i] * (inp[i] * inv);
    }
}

fn silu(input: []f32, output: []f32) void {
    var i: usize = 0;
    while (i < input.len) : (i += 1) {
        const x = input[i];
        output[i] = x / (1.0 + math.exp(-x));
    }
}

fn accumulate(out: []f32, inp: []const f32) void {
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        out[i] += inp[i];
    }
}

fn applySoftmax(att: []f32) void {
    var max_val = -math.inf(f32);
    for (att) |v| {
        if (v > max_val) max_val = v;
    }
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < att.len) : (i += 1) {
        const e = math.exp(att[i] - max_val);
        att[i] = e;
        sum += e;
    }
    const inv_sum = 1.0 / sum;
    i = 0;
    while (i < att.len) : (i += 1) {
        att[i] *= inv_sum;
    }
}

pub fn applyAttention(
    config: Config,
    state: *RunState,
    pos: usize,
    head_scale: f32,
    key_layer: []f32,
    value_layer: []f32,
) void {
    const head_size = config.headSize();
    const kv_dim = config.kvDim();
    const n_heads = config.n_heads;
    const n_kv_heads = config.n_kv_heads;
    const kv_mul = n_heads / n_kv_heads;

    ropeRotate(state.q, head_size, pos, state.rope_cache);
    const k_here = key_layer[pos * kv_dim .. (pos + 1) * kv_dim];
    ropeRotate(k_here, head_size, pos, state.rope_cache);

    var head: usize = 0;
    while (head < n_heads) : (head += 1) {
        const head_offset = head * head_size;
        const att_head = state.att[head * config.seq_len .. (head + 1) * config.seq_len];
        var t: usize = 0;
        const kv_head = head / kv_mul;
        while (t <= pos) : (t += 1) {
            const key_t_offset = t * kv_dim + kv_head * head_size;
            var score: f32 = 0.0;
            var i: usize = 0;
            while (i < head_size) : (i += 1) {
                score += state.q[head_offset + i] * key_layer[key_t_offset + i];
            }
            att_head[t] = score * head_scale;
        }
        applySoftmax(att_head[0 .. pos + 1]);
        var i_dim: usize = 0;
        while (i_dim < head_size) : (i_dim += 1) {
            var acc: f32 = 0.0;
            var t_att: usize = 0;
            while (t_att <= pos) : (t_att += 1) {
                const value_t_offset = t_att * kv_dim + kv_head * head_size;
                acc += att_head[t_att] * value_layer[value_t_offset + i_dim];
            }
            state.xb[head_offset + i_dim] = acc;
        }
    }
}

pub fn forward(
    config: Config,
    weights: TransformerWeights,
    state: *RunState,
    token: usize,
    pos: usize,
) void {
    const dim = config.dim;
    const kv_dim = config.kvDim();
    const head_size = config.headSize();
    std.mem.copy(f32, state.x, weights.token_embedding_table[token * dim .. (token + 1) * dim]);

    const head_scale = 1.0 / math.sqrt(@as(f32, @floatFromInt(head_size)));

    var layer: usize = 0;
    while (layer < config.n_layers) : (layer += 1) {
        const layer_offset = layer * dim;
        rmsNorm(state.xb, state.x, weights.rms_att_weight[layer_offset .. layer_offset + dim]);

        const wq_slice = weights.wq[layer * dim * dim .. (layer + 1) * dim * dim];
        const wk_slice = weights.wk[layer * dim * kv_dim .. (layer + 1) * dim * kv_dim];
        const wv_slice = weights.wv[layer * dim * kv_dim .. (layer + 1) * dim * kv_dim];
        const layer_cache_offset = layer * config.seq_len * kv_dim;
        var key_layer = state.key_cache[layer_cache_offset .. layer_cache_offset + config.seq_len * kv_dim];
        var value_layer = state.value_cache[layer_cache_offset .. layer_cache_offset + config.seq_len * kv_dim];
        var k_here = key_layer[pos * kv_dim .. (pos + 1) * kv_dim];
        var v_here = value_layer[pos * kv_dim .. (pos + 1) * kv_dim];
        matmul_simd(state.q, state.xb, wq_slice, dim, dim);
        matmul_simd(k_here, state.xb, wk_slice, kv_dim, dim);
        matmul_simd(v_here, state.xb, wv_slice, kv_dim, dim);

        applyAttention(config, state, pos, head_scale, key_layer, value_layer);

        const wo_slice = weights.wo[layer * dim * dim .. (layer + 1) * dim * dim];
        matmul_simd(state.xb2, state.xb, wo_slice, dim, dim);
        accumulate(state.x, state.xb2);

        const hidden = config.hidden_dim;
        rmsNorm(state.xb, state.x, weights.rms_ffn_weight[layer_offset .. layer_offset + dim]);
        const w1_slice = weights.w1[layer * dim * hidden .. (layer + 1) * dim * hidden];
        const w3_slice = weights.w3[layer * dim * hidden .. (layer + 1) * dim * hidden];
        matmul_simd(state.hb, state.xb, w1_slice, hidden, dim);
        matmul_simd(state.hb2, state.xb, w3_slice, hidden, dim);
        silu(state.hb, state.hb);
        var i: usize = 0;
        while (i < state.hb.len) : (i += 1) {
            state.hb[i] *= state.hb2[i];
        }
        const w2_slice = weights.w2[layer * hidden * dim .. (layer + 1) * hidden * dim];
        matmul_simd(state.xb2, state.hb, w2_slice, dim, hidden);
        accumulate(state.x, state.xb2);
    }

    rmsNorm(state.x, state.x, weights.rms_final_weight);
    matmul_simd(state.logits, state.x, weights.wcls, config.vocab_size, dim);
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Zig port placeholder.\n", .{});
}
