# Predicted outputs and the recall store (text for the README)

## `prediction`: tell the server what the reply will mostly repeat

When a reply will repeat text the client already has (a file being rewritten, a tool call that writes a whole file, a document being
re-generated with a small change), send that text with the request. The server uses it only as a source of **drafts**: every drafted
token is checked by the model and kept only if it is the token the model would have produced anyway. The reply is the same with or
without the field; only the speed changes. A wrong prediction costs about nothing (measured -0.2 % on chat).

The field has the shape of OpenAI's predicted outputs and is accepted on all three routes:

```jsonc
// POST /v1/chat/completions, POST /v1/responses, POST /v1/messages
{
  "model": "Qwen3.8-Flash-Next",
  "messages": [...],                       // or "input" on /v1/responses
  "prediction": {"type": "content", "content": "the text the reply is expected to repeat"}
}
```

`content` may also be a list of parts, `[{"type": "text", "text": "..."}, ...]`; parts are kept apart (a copy never runs from one part
into the next), so an agent harness can send several files. Which text to send:

| request | send as prediction |
|---|---|
| rewrite / edit a file whose text the client holds | the current file |
| agent turn that may call `write_file` or `str_replace` | the files read or written so far in the session (raw text, not JSON-escaped: this model writes tool-call values raw) |
| the prompt holds a numbered listing (`cat -n`) and the reply is the plain file | the plain file |
| "the same again, but change X" | the earlier reply |

Rules: the field is optional; an unknown shape, a non-string, an empty or an oversized prediction is ignored, never refused;
`"draft": false` turns it off together with all other drafts; sampled requests keep their per-seed output. Server knobs:
`TF_Q38_PREDICTION=0` ignores the field, `TF_Q38_PRED_MAX` caps its tokens (65536), `TF_Q38_PRED_MATCH` is the match a copy needs (8).

Where it does nothing: chat and any reply that does not repeat text; replies whose text is already in the prompt word for word gain
little over the prompt copy the engine does by itself.

## `--recall`: let later requests copy from earlier ones (opt-in, single-user boxes)

`serve ... --recall --recall-dir DIR [--recall-gib G] [--recall-read-only]` keeps a record of the prompts and replies the server has seen
and drafts copies from it (TensorFold's own feature). Again drafts only: replies do not change.

- The record is **shared by every request the server gets and is written to disk** in `DIR`. On a box with one user that is the point
  (a second agent session on the same repository starts faster). On a shared box it means one user's text can be proposed as a draft
  in another user's request; it can only be *kept* there if the model would have written the same tokens anyway, so no reply changes,
  but timing differs with what others sent, and the text sits in `DIR`. Do not turn it on for a multi-tenant server.
- Bounds: `--recall-gib` caps the disk (default 1/16 GiB; RAM about twice that when full); `GET /v1/recall` shows its state,
  `DELETE /v1/recall` empties it.
- Per request: `"recall": false` keeps a request out of the record (it neither reads it nor is added to it).
