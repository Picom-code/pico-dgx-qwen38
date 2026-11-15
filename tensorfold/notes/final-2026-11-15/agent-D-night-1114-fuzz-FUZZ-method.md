## Method (what was compared with what)

- Binary under test: the final image's engine (tensorfold-qwen38:v1.1.4-eab8582, patches 0101-0129), copied from spark-923b to the head,
  tensorfold-native sha256 af27a1e7b029e243 (116 files, 0 sha mismatches). KV cache bf16. Head launcher = the environment recipe.sh computes
  (recenv.sh) with 2 x 262K lanes instead of 3 (the head keeps a 13 GiB reserve for DeepSeek's rank 0, the gateway and the vision encoder).
- Reference = the SAME running engine with every lever that has a knob-file word switched off (23 words: cert, cert_topk, cert_gather, hcf,
  half_below, mtp_last, gdn_many, best_lanes, attn_lanes, late_skip, keep_inpass, lane_reuse, head2, head2_sampled, deep_off, stop2,
  ngram_split, qsa_cert, qsa_list, qsa_rows, qsa_cert_min, exp_half, prefix_wait_ms). Every request runs off / on back to back (order rotated),
  and off a second time: a request whose two levers-off replies differ is "base not deterministic" and is not counted for or against a lever.
- Levers WITHOUT a knob word (128-row copy windows + COPY_RESET + COPY_SHARED, PREFILL_MANY, elastic prompt cache, delta prompt states,
  prompt-pass levers GDN_PROD / ROUTE_BAND / SCRATCH_TRIM + CHUNK / HC_DN_TALL / HEAD_KV / WARM, QSA_HALF, HEAD2_FAST, CERT_PLAIN, CERT_CODE,
  DRAFTS=12) stay ON in the knob-off reference. They are covered by a SECOND ENGINE START of the same binary with every lever off by its
  flag (recipe options CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0
  LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0 PREFIX_WAIT=0, TF_FLASHNEXT_Q38_PREDICTION=0): its replies are compared with the knob-off replies.
- The `prediction` field: the reference is the same request WITHOUT the field (levers off).
- Compared per reply: HTTP status, error text, answer text, reasoning text, tool calls (name + arguments as canonical JSON), finish reason,
  prompt tokens, completion tokens (prompt tokens are not compared on /v1/messages: Anthropic accounting leaves cached tokens out).
- Streaming: the concatenated chunks of a streamed reply are compared with the streamed levers-off reply (and with the non-streamed one).
- In flight: each reply of a run with N requests in flight is compared with the SOLO levers-off reply of the same request.
- Sessions: live multi-turn conversations (each real reply goes back as the assistant turn), with "regenerate" and "edit an earlier turn"
  steps, under levers off / on / on after a client-cancelled stream; then every turn's request is replayed COLD (fresh flags-off engine,
  last turn first) and compared with the resumed reply.
- Tools: Mac q38/kitD/redteam (rt.py: gen / gen2 / gen3 / pair / pairsum / sess / run / cmp / bisect / probe; fuzzrep.py: the tables;
  rtq.sh / rtchain.sh / run12.sh / rj.sh: head queue with the window cap and the 12:04 UTC cutoff; stub.py: dry-check engine).
