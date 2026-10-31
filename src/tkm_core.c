// SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// Copyright (c) 2026 BSODsystem32

/*
 * src/tkm_core.c -- Table-Driven Parser (with `block x N:` Loop Macro), Fuser & Arena Planner
 */
#include "tkm.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define TKM_ROWS_PER_CTA 16

static void skip_ws(const char **p) {
    while (**p) {
        if (isspace((unsigned char)**p)) (*p)++;
        else if (**p == '#') { while (**p && **p != '\n') (*p)++; }
        else break;
    }
}

static int match_kw(const char **p, const char *kw) {
    skip_ws(p); size_t len = strlen(kw);
    if (strncmp(*p, kw, len) == 0 && !isalnum((unsigned char)(*p)[len]) && (*p)[len] != '_') {
        *p += len; return 1;
    }
    return 0;
}

static void read_ident(const char **p, char *out, int max_len) {
    skip_ws(p); int i = 0;
    while (**p && (isalnum((unsigned char)**p) || **p == '_')) {
        if (i < max_len - 1) out[i++] = **p;
        (*p)++;
    }
    out[i] = '\0';
}

static float read_num(const char **p) {
    skip_ws(p); char *end; float v = strtof(*p, &end); *p = end; return v;
}

static int new_node(TkmGraph *g, const TkmOpSpec *spec, int in0, int in1, int in2, const char *tag) {
    int id = g->num_nodes++;
    TkmNode *n = &g->nodes[id];
    memset(n, 0, sizeof(TkmNode));
    n->id = id;
    n->op = spec ? spec->op : OP_INPUT;
    n->cat = spec ? spec->cat : CAT_META;
    n->in0 = in0; n->in1 = in1; n->in2 = in2;
    n->plasma_grp = -1; n->out_buf_id = -1;
    n->k = 1; n->s = 1; n->p = 0;
    n->fval = spec ? spec->default_fval : 0.0f;
    n->fval2 = 4.0f;
    n->rng_type = RNG_DEFAULT;
    snprintf(n->tag, sizeof(n->tag), "%s_%d", tag ? tag : (spec ? spec->dsl_name : "input"), id);
    return id;
}

static void parse_args(const char **p, TkmNode *n) {
    skip_ws(p);
    if (**p != '(') return;
    (*p)++;
    int pos = 0;
    while (**p && **p != ')') {
        skip_ws(p);
        const char *save = *p;
        char key[32];
        read_ident(p, key, sizeof(key));
        skip_ws(p);
        if (**p == '=') {
            (*p)++;
            if (!strcmp(key, "prng") || !strcmp(key, "rng")) {
                char rname[32];
                read_ident(p, rname, sizeof(rname));
                n->rng_type = tkm_parse_rng_name(rname);
            } else {
                float v = read_num(p);
                if (!strcmp(key, "c") || !strcmp(key, "out")) n->c_out = (int)v;
                else if (!strcmp(key, "k")) n->k = (int)v;
                else if (!strcmp(key, "s")) n->s = (int)v;
                else if (!strcmp(key, "p")) n->p = (int)v;
                else if (!strcmp(key, "q")) n->q_heads = (int)v;
                else if (!strcmp(key, "kv")) n->kv_heads = (int)v;
                else if (!strcmp(key, "d")) n->head_dim = (int)v;
                else if (!strcmp(key, "r") || !strcmp(key, "sigma") || !strcmp(key, "radius")) n->fval2 = v;
                else n->fval = v;
            }
        } else {
            *p = save;
            float v = read_num(p);
            if (pos == 0) {
                if (n->op == OP_FC || n->op == OP_CONV) n->c_out = (int)v;
                else if (n->op == OP_MIXPOOL || n->op == OP_AVGPOOL) n->k = (int)v;
                else n->fval = v;
            } else if (pos == 1) {
                n->fval2 = v;
            }
            pos++;
        }
        skip_ws(p);
        if (**p == ',') (*p)++;
    }
    if (**p == ')') (*p)++;
}

static int parse_chain(TkmGraph *g, const char **p, int cur_in, const char *tag);

static int parse_step(TkmGraph *g, const char **p, int cur_in, const char *tag) {
    skip_ws(p);
    if (**p == '{') {
        (*p)++;
        int ends[8], nb = 0;
        while (**p && **p != '}') {
            skip_ws(p); if (**p == '}') break;
            char btag[32] = "br";
            const char *save = *p;
            read_ident(p, btag, sizeof(btag));
            skip_ws(p);
            if (**p == ':') (*p)++; else *p = save;
            ends[nb++] = parse_chain(g, p, cur_in, btag);
            skip_ws(p);
            if (**p == ',' || **p == ';') (*p)++;
        }
        if (**p == '}') (*p)++;
        skip_ws(p);
        if ((*p)[0] == '-' && (*p)[1] == '>') *p += 2;
        char mop_s[32];
        read_ident(p, mop_s, sizeof(mop_s));
        const TkmOpSpec *mspec = tkm_find_spec_by_name(mop_s);
        if (!mspec) { fprintf(stderr, "Unknown merge op '%s'\n", mop_s); exit(1); }
        int nid = new_node(g, mspec, nb > 0 ? ends[0] : -1, nb > 1 ? ends[1] : -1, nb > 2 ? ends[2] : -1, tag);
        parse_args(p, &g->nodes[nid]);
        return nid;
    }

    char op_s[32];
    read_ident(p, op_s, sizeof(op_s));
    if (!op_s[0]) return cur_in;
    if (!strcmp(op_s, "Identity")) return cur_in;

    const TkmOpSpec *spec = tkm_find_spec_by_name(op_s);
    if (!spec) {
        fprintf(stderr, "[TKM Error] Unknown operator '%s' (Did you register it in tkm_registry.c?)\n", op_s);
        exit(1);
    }

    int nid = new_node(g, spec, cur_in, -1, -1, tag);
    parse_args(p, &g->nodes[nid]);
    if (spec->op == OP_CONV || spec->op == OP_FC) {
        return new_node(g, tkm_find_spec_by_op(OP_BIAS), nid, -1, -1, tag);
    }
    return nid;
}

static int parse_chain(TkmGraph *g, const char **p, int cur_in, const char *tag) {
    int cur = parse_step(g, p, cur_in, tag);
    while (1) {
        skip_ws(p);
        if ((*p)[0] == '-' && (*p)[1] == '>') {
            *p += 2;
            cur = parse_step(g, p, cur, tag);
        } else break;
    }
    return cur;
}

static int parse_statement_block(TkmGraph *g, const char **p, int cur_tail, int stop_at_brace) {
    while (**p) {
        skip_ws(p);
        if (!**p) break;
        if (stop_at_brace && **p == '}') {
            (*p)++;
            break;
        }
        if (match_kw(p, "input")) {
            skip_ws(p); if (**p == ':') (*p)++;
            int d0 = (int)read_num(p);
            skip_ws(p); if (**p == 'x' || **p == 'X') (*p)++;
            int d1 = (int)read_num(p);
            int d2 = 1;
            skip_ws(p); if (**p == 'x' || **p == 'X') { (*p)++; d2 = (int)read_num(p); }
            cur_tail = new_node(g, NULL, -1, -1, -1, "input");
            g->nodes[cur_tail].H = d0; g->nodes[cur_tail].W = d1; g->nodes[cur_tail].C = d2;
            continue;
        }
        if (match_kw(p, "prng") || match_kw(p, "rng")) {
            skip_ws(p); if (**p == ':') (*p)++;
            char rname[32];
            read_ident(p, rname, sizeof(rname));
            g->default_rng = tkm_parse_rng_name(rname);
            continue;
        }
        if (match_kw(p, "flow")) {
            skip_ws(p); if (**p == ':') (*p)++;
            continue;
        }

        char label[32];
        read_ident(p, label, sizeof(label));
        if (!label[0]) break; /* Safely break on trailing whitespace / EOF */

        skip_ws(p);
        int repeat_cnt = 1;
        if (**p == 'x' || **p == 'X' || **p == '*') {
            const char *save = *p;
            (*p)++;
            skip_ws(p);
            if (isdigit((unsigned char)**p)) {
                repeat_cnt = (int)read_num(p);
            } else {
                *p = save;
            }
        }
        skip_ws(p);
        if (**p == ':') (*p)++;
        skip_ws(p);

        if (repeat_cnt > 1 && **p == '{') {
            (*p)++;
            const char *body_start = *p;
            for (int r = 0; r < repeat_cnt; r++) {
                const char *iter_p = body_start;
                cur_tail = parse_statement_block(g, &iter_p, cur_tail, 1);
                if (r == repeat_cnt - 1) *p = iter_p;
            }
        } else {
            cur_tail = parse_chain(g, p, cur_tail, label);
        }
    }
    return cur_tail;
}

void tkm_parse_dsl(TkmGraph *g, const char *dsl_text, int batch_size) {
    memset(g, 0, sizeof(TkmGraph));
    g->batch_size  = batch_size;
    g->default_rng = RNG_WANG_XORSHIFT32;
    const char *p  = dsl_text;
    parse_statement_block(g, &p, -1, 0);

    for (int i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].rng_type == RNG_DEFAULT) g->nodes[i].rng_type = g->default_rng;
    }
}

void tkm_infer_shapes(TkmGraph *g) {
    for (int i = 0; i < g->num_nodes; i++) {
        TkmNode *n = &g->nodes[i];
        if (n->dead || n->op == OP_INPUT) continue;
        TkmNode *in0 = &g->nodes[n->in0];
        n->H = in0->H; n->W = in0->W; n->C = in0->C;
        switch (n->op) {
            case OP_CONV: n->H = (in0->H + 2*n->p - n->k)/n->s + 1; n->W = (in0->W + 2*n->p - n->k)/n->s + 1; n->C = n->c_out; break;
            case OP_MIXPOOL: case OP_AVGPOOL: n->H = in0->H / n->k; n->W = in0->W / n->k; break;
            case OP_FLATTEN: n->C = in0->H * in0->W * in0->C; n->H = 1; n->W = 1; break;
            case OP_FC: case OP_FC_TWIN_FUSED: case OP_FC_QKV_FUSED: n->C = n->c_out; break;
            case OP_QKNORM: if (n->c_out > 0) n->C = n->c_out; break;
            case OP_MUL: if (n->c_out > 0) n->C = n->c_out; break;
            case OP_GQA_ATTN: n->C = n->q_heads * n->head_dim; break;
            case OP_ADD: {
                TkmNode *in1 = &g->nodes[n->in1];
                if (in0->H != in1->H || in0->W != in1->W || in0->C != in1->C) {
                    fprintf(stderr, "[TKM Shape Error] Node #%d (%s) branch mismatch: [%dx%dx%d] vs [%dx%dx%d]\n",
                            i, n->tag, in0->H, in0->W, in0->C, in1->H, in1->W, in1->C);
                    exit(1);
                }
                break;
            }
            default: break;
        }
    }
}

static void add_karg(TkmKernelSig *sig, TkmArgKind kind, int nid, int append_nid, const char *ctype, const char *prefix) {
    TkmKernelArg *a = &sig->args[sig->num_args++];
    a->kind = kind; a->node_id = nid;
    snprintf(a->c_type, sizeof(a->c_type), "%s", ctype);
    if (append_nid) snprintf(a->c_name, sizeof(a->c_name), "%s_%d", prefix, nid);
    else            snprintf(a->c_name, sizeof(a->c_name), "%s", prefix);
}

static void bind_node_fwd_resources(TkmKernelSig *fs, const TkmNode *n) {
    const TkmOpSpec *spec = tkm_find_spec_by_op(n->op);
    if (!spec) return;
    uint32_t f = spec->res_flags;
    if (f & RES_PARAM_BIAS)   add_karg(fs, KARG_BIAS,        n->id,  1, "const float2* __restrict__", "b_vec");
    if (f & RES_PARAM_WEIGHT) add_karg(fs, KARG_WEIGHT_VEC,  n->id,  1, "const float2* __restrict__", "w_vec");
    if (f & RES_PARAM_BETA)   add_karg(fs, KARG_BETA,        n->id,  1, "const float2* __restrict__", "beta_vec");
    if (f & RES_SAVE_ISTD)    add_karg(fs, KARG_SAVE_ISTD,   n->id,  1, "float* __restrict__",        "save_istd");
    if (f & RES_SAVE_XHAT)    add_karg(fs, KARG_SAVE_XHAT,   n->id,  1, "__half2* __restrict__",      "save_xhat");
    if (f & RES_SAVE_MASK)    add_karg(fs, KARG_SAVE_MASK,   n->id,  1, "uint16_t* __restrict__",     "save_mask");
    if (f & RES_RNG_SEED)     add_karg(fs, KARG_STEP_SEED,   n->id,  1, "const uint32_t* __restrict__","d_seed");
    if (f & RES_BRANCH_IN)    add_karg(fs, KARG_BRANCH_IN,   n->in1, 1, "const __half2* __restrict__", "branch_in");
}

static void bind_node_bwd_resources(TkmKernelSig *bs, const TkmNode *n) {
    const TkmOpSpec *spec = tkm_find_spec_by_op(n->op);
    if (!spec) return;
    uint32_t f = spec->res_flags;
    if (f & RES_BRANCH_IN)    add_karg(bs, KARG_DBRANCH_OUT, n->in1, 1, "__half2* __restrict__",       "dbranch_out");
    if (f & RES_SAVE_MASK)    add_karg(bs, KARG_SAVE_MASK,   n->id,  1, "const uint16_t* __restrict__", "save_mask");
    if (f & RES_SAVE_XHAT)    add_karg(bs, KARG_SAVE_XHAT,   n->id,  1, "const __half2* __restrict__",  "save_xhat");
    if (f & RES_SAVE_ISTD)    add_karg(bs, KARG_SAVE_ISTD,   n->id,  1, "const float* __restrict__",    "save_istd");
    if (f & RES_PARAM_WEIGHT) {
        add_karg(bs, KARG_WEIGHT_VEC,  n->id, 1, "const float2* __restrict__", "w_vec");
        add_karg(bs, KARG_DWEIGHT_VEC, n->id, 1, "float* __restrict__",        "dw_vec");
    }
    if (f & RES_PARAM_BETA) {
        add_karg(bs, KARG_BETA,  n->id, 1, "const float2* __restrict__", "beta_vec");
        add_karg(bs, KARG_DBETA, n->id, 1, "float* __restrict__",        "dbeta_vec");
    }
    if (f & RES_PARAM_BIAS) {
        add_karg(bs, KARG_BIAS,  n->id, 1, "const float2* __restrict__", "b_vec");
        add_karg(bs, KARG_DBIAS, n->id, 1, "float* __restrict__",        "db_vec");
    }
}

static void build_group_abi(const TkmGraph *g, PlasmaGroup *pg) {
    TkmKernelSig *fs = &pg->fwd_sig, *bs = &pg->bwd_sig;
    memset(fs, 0, sizeof(*fs)); memset(bs, 0, sizeof(*bs));
    snprintf(fs->func_name, sizeof(fs->func_name), "tkm_plasma_fwd_g%d", pg->id);
    snprintf(bs->func_name, sizeof(bs->func_name), "tkm_plasma_bwd_g%d", pg->id);

    if (pg->is_coord_group) {
        fs->grid_x  = bs->grid_x  = g->batch_size;
        fs->block_x = bs->block_x = 256;
        add_karg(fs, KARG_IN_MAIN,   pg->anchor_id,   0, "const __half* __restrict__",   "in_img");
        add_karg(fs, KARG_OUT_MAIN,  pg->out_node_id, 0, "__half* __restrict__",         "out_img");
        add_karg(fs, KARG_STEP_SEED, pg->out_node_id, 0, "const uint32_t* __restrict__", "d_seed");

        add_karg(bs, KARG_DOUT_MAIN, pg->out_node_id, 0, "const __half* __restrict__",   "dout_img");
        add_karg(bs, KARG_DIN_MAIN,  pg->anchor_id,   0, "__half* __restrict__",         "din_img");
        return;
    }

    if (pg->C % 2 != 0) {
        fprintf(stderr, "[TKM Shape Error] PlasmaGroup #%d has odd channel count %d; CUDA half2 kernels require an even channel count.\n",
                pg->id, pg->C);
        exit(1);
    }

    int rows = g->batch_size * pg->H * pg->W;
    int C2 = pg->C / 2;
    int rows_per_blk = (!pg->is_qkv_prep && C2 > 0 && C2 <= 32 && (256 % C2 == 0)) ? (256 / C2) : TKM_ROWS_PER_CTA;
    fs->grid_x  = bs->grid_x  = (rows + rows_per_blk - 1) / rows_per_blk;
    fs->block_x = bs->block_x = 256;

    add_karg(fs, KARG_IN_MAIN,   pg->anchor_id,   0, "const __half2* __restrict__", "in_0");
    add_karg(fs, KARG_OUT_MAIN,  pg->out_node_id, 0, "__half2* __restrict__",       "out_0");

    add_karg(bs, KARG_DOUT_MAIN, pg->out_node_id, 0, "const __half2* __restrict__", "dout_0");
    add_karg(bs, KARG_IN_MAIN,   pg->anchor_id,   0, "const __half2* __restrict__", "replay_in_0");
    add_karg(bs, KARG_DIN_MAIN,  pg->anchor_id,   0, "__half2* __restrict__",       "din_0");

    if (pg->is_twin_epilogue) {
        const TkmNode *mul_n = &g->nodes[pg->node_ids[0]];
        for (int i = 0; i < mul_n->left_len; i++)  bind_node_fwd_resources(fs, &g->nodes[mul_n->left_nodes[i]]);
        for (int i = 0; i < mul_n->right_len; i++) bind_node_fwd_resources(fs, &g->nodes[mul_n->right_nodes[i]]);
        for (int i = 0; i < mul_n->left_len; i++)  bind_node_bwd_resources(bs, &g->nodes[mul_n->left_nodes[i]]);
        for (int i = 0; i < mul_n->right_len; i++) bind_node_bwd_resources(bs, &g->nodes[mul_n->right_nodes[i]]);
    }
    for (int i = 0; i < pg->count; i++) bind_node_fwd_resources(fs, &g->nodes[pg->node_ids[i]]);
    for (int i = pg->count - 1; i >= 0; i--) bind_node_bwd_resources(bs, &g->nodes[pg->node_ids[i]]);
}

void tkm_optimize_and_bind(TkmGraph *g) {
    int num_rules = 0;
    const TkmFusionRule *rules = tkm_get_fusion_rules(&num_rules);

    tkm_infer_shapes(g);

    for (int i = 0; i < g->num_nodes; i++) {
        const TkmNode *n = &g->nodes[i];
        if (!n->dead && n->op == OP_MIXPOOL && n->C % 2 != 0) {
            fprintf(stderr, "[TKM Shape Error] MixPool node #%d (%s) has odd channel count %d; CUDA half2 kernels require an even channel count.\n",
                    i, n->tag, n->C);
            exit(1);
        }
    }

    for (int i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].dead) continue;
        for (int r = 0; r < num_rules; r++) {
            if (rules[r].match(g, i)) {
                rules[r].apply(g, i);
                break;
            }
        }
    }

    tkm_infer_shapes(g);

    for (int i = 0; i < g->num_nodes; i++) g->nodes[i].users = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].dead) continue;
        if (g->nodes[i].in0 >= 0) g->nodes[g->nodes[i].in0].users++;
        if (g->nodes[i].in1 >= 0) g->nodes[g->nodes[i].in1].users++;
        if (g->nodes[i].in2 >= 0) g->nodes[g->nodes[i].in2].users++;
    }

    g->num_groups = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        TkmNode *n = &g->nodes[i];
        if (n->dead || n->plasma_grp >= 0) continue;

        if (n->cat == CAT_COORD) {
            int gid = g->num_groups++;
            PlasmaGroup *pg = &g->groups[gid];
            memset(pg, 0, sizeof(*pg));
            pg->id = gid; pg->anchor_id = n->in0;
            pg->H = n->H; pg->W = n->W; pg->C = n->C;
            pg->is_coord_group = 1;

            int cur = i, seen_pixel_op = 0;
            while (cur >= 0 && cur < g->num_nodes) {
                TkmNode *cn = &g->nodes[cur];
                if (cn->dead || cn->plasma_grp >= 0 || cn->cat != CAT_COORD) break;
                int is_pix = (cn->op == OP_RAND_NOISE || cn->op == OP_RAND_CUTOUT ||
                              cn->op == OP_RAND_BRIGHTNESS || cn->op == OP_RAND_CONTRAST);
                if (seen_pixel_op && !is_pix) break;
                if (is_pix) seen_pixel_op = 1;

                pg->node_ids[pg->count++] = cur;
                pg->out_node_id = cur;
                cn->plasma_grp = gid;
                if (cn->users > 1) break;
                int next_id = -1;
                for (int k = cur + 1; k < g->num_nodes; k++) {
                    if (!g->nodes[k].dead && g->nodes[k].in0 == cur) { next_id = k; break; }
                }
                cur = next_id;
            }
            printf("  [Rule Hit: Fused 2D Coordinate+Pixel Pipeline] %d bricks -> PlasmaGroup #%d (PRNG=%s)\n",
                   pg->count, gid, tkm_rng_name(n->rng_type));
            build_group_abi(g, pg);
            continue;
        }

        if (n->op == OP_QKNORM && n->q_c > 0) {
            int gid = g->num_groups++;
            PlasmaGroup *pg = &g->groups[gid];
            memset(pg, 0, sizeof(*pg));
            pg->id = gid; pg->anchor_id = n->in0;
            pg->H = n->H; pg->W = n->W; pg->C = n->C;
            pg->is_qkv_prep = 1;
            pg->node_ids[pg->count++] = i;
            pg->out_node_id = i;
            n->plasma_grp = gid;
            build_group_abi(g, pg);
            continue;
        }

        if (n->cat == CAT_NORM || n->cat == CAT_FLUID) {
            int gid = g->num_groups++;
            PlasmaGroup *pg = &g->groups[gid];
            memset(pg, 0, sizeof(*pg));
            pg->id = gid; pg->anchor_id = n->in0;
            pg->H = n->H; pg->W = n->W; pg->C = n->C;
            if (n->op == OP_MUL && n->left_len > 0) pg->is_twin_epilogue = 1;

            int cur = i;
            while (cur >= 0 && cur < g->num_nodes) {
                TkmNode *cn = &g->nodes[cur];
                if (cn->dead || cn->plasma_grp >= 0) break;
                if (cn->cat != CAT_NORM && cn->cat != CAT_FLUID) break;
                if (cn->cat == CAT_NORM && pg->has_norm) break;

                pg->node_ids[pg->count++] = cur;
                pg->out_node_id = cur;
                cn->plasma_grp = gid;
                if (cn->cat == CAT_NORM) pg->has_norm = 1;
                if (cn->users > 1) break;

                int next_id = -1;
                for (int k = cur + 1; k < g->num_nodes; k++) {
                    if (!g->nodes[k].dead && g->nodes[k].in0 == cur) { next_id = k; break; }
                }
                cur = next_id;
            }
            build_group_abi(g, pg);
        }
    }
}

void tkm_plan_memory(TkmGraph *g) {
    int num_slots = 0;
    size_t peak_bytes = 0, naive_bytes = 0;

    for (int i = 0; i < g->num_nodes; i++) {
        TkmNode *n = &g->nodes[i];
        if (n->dead) continue;

        size_t bytes = (size_t)g->batch_size * n->H * n->W * n->C * 2;
        bytes = (bytes + 255) & ~(size_t)255;
        naive_bytes += bytes;

        if (n->plasma_grp >= 0 && g->groups[n->plasma_grp].out_node_id != i) {
            n->out_buf_id = -1;
            n->out_bytes = 0;
            continue;
        }

        n->out_bytes  = bytes;
        n->out_buf_id = num_slots++;
        n->out_offset = peak_bytes;
        peak_bytes   += bytes;
    }
    g->total_arena_bytes = peak_bytes;
    g->naive_arena_bytes = naive_bytes;
}

void tkm_dump_summary(const TkmGraph *g, FILE *out) {
    double naive_mb = g->naive_arena_bytes / (1024.0 * 1024.0);
    double peak_mb  = g->total_arena_bytes / (1024.0 * 1024.0);
    double saved    = 100.0 * (1.0 - peak_mb / (naive_mb > 0 ? naive_mb : 1.0));

    fprintf(out, "  [Default PRNG ] %s\n", tkm_rng_name(g->default_rng));
    fprintf(out, "  [Arena Planner] Unfused Activation Memory: %.2f MB -> Fused Arena: %.2f MB (Saved %.1f%%)\n\n",
            naive_mb, peak_mb, saved);
    fprintf(out, " ID | %-18s | %-18s | In:(0,1) | Out Shape     | Memory Confinement\n", "Tag", "OpType");
    fprintf(out, "----+--------------------+--------------------+----------+---------------+----------------------------------\n");
    for (int i = 0; i < g->num_nodes; i++) {
        const TkmNode *n = &g->nodes[i];
        if (n->dead) continue;
        const TkmOpSpec *spec = tkm_find_spec_by_op(n->op);
        const char *op_s = spec ? spec->dsl_name : "Input";
        char mem_s[96];
        if (n->out_buf_id < 0) {
            snprintf(mem_s, sizeof(mem_s), "Register-Only (Plasma #%d)", n->plasma_grp);
        } else if (n->plasma_grp >= 0) {
            snprintf(mem_s, sizeof(mem_s), "Slot #%d (Off:%5zuKB) [Plasma #%d Out]", n->out_buf_id, n->out_offset / 1024, n->plasma_grp);
        } else {
            snprintf(mem_s, sizeof(mem_s), "Slot #%d (Off:%5zuKB) [Anchor Out]", n->out_buf_id, n->out_offset / 1024);
        }
        fprintf(out, " #%2d| %-18s | %-18s | (%2d,%3d) | [%2dx%3dx%4d] | %s\n",
                n->id, n->tag, op_s, n->in0, n->in1, n->H, n->W, n->C, mem_s);
    }
    fprintf(out, "\n");
}

int tkm_verify_registry(void) {
    int errors = 0;
    for (int i = 0; i < OP_COUNT; i++) {
        const TkmOpSpec *spec = tkm_find_spec_by_op((TkmOpType)i);
        if (!spec) {
            fprintf(stderr, "[Registry Error] Op enum %d has no entry in TKM_OP_REGISTRY[]!\n", i);
            errors++;
            continue;
        }
        if (spec->cat == CAT_NORM && !(spec->res_flags & RES_SAVE_XHAT)) {
            fprintf(stderr, "[Registry Error] Normalization op '%s' must declare RES_SAVE_XHAT!\n", spec->dsl_name);
            errors++;
        }
        if (spec->cat == CAT_FLUID && spec->fwd_cuda && !spec->bwd_cuda) {
            fprintf(stderr, "[Registry Error] Operator '%s' has fwd_cuda but missing bwd_cuda!\n", spec->dsl_name);
            errors++;
        }
    }
    return errors;
}