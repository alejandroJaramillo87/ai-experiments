
● Phase 1 is complete. make quality exits 0; tests/models is 106 passed + 1 xfailed, up from 58.

  What the mechanism found

  The plan assumed the facts were right and only unguarded. Building the guard turned up two real defects in the committed evidence:

  All 14 REFUSED verdicts in gemma4-gguf.txt are false. gguf_header_transcript.sh carried hand-copied type sets — {0,1,8} for tensors, {5,6,8} for array elements — with line
  citations that had already drifted. tensor.rs had since gained BF16 (type 30) and gguf.rs had gained bool arrays (type 7). So the transcript reported BF16 (type 30): 318 
  tensors -- REFUSED against the E2B checkpoint the harness exists to load, and flagged sliding_window_pattern as unparseable. A third copy of the same sets sat in
  specbuild.py, and four tests asserted the false verdicts were correct. All three copies are now derived from the Rust source; the tests were rewritten to keep their
  negative-control value using types the loader genuinely rejects (Q4_K, uint32 arrays).

  The 12B/31B tensor shapes I derived are absent from the contract by decision. Config implies num_global_key_value_heads: 1 on 12B, giving a full-attention k_proj narrower
  than the sliding one — plausible but unverifiable without headers, so recording it would have been a derivation presented as an observation.

  What was verified, not assumed

  All four config.json files fetched at their pinned revisions and every git blob hash matched the doc's pins. Every value in gemma-4.md:107-176 matches those bytes exactly,
  including all four full-attention index arrays. The E2B/E4B projection shapes I derived from config match the doc's safetensors-sourced table exactly — two independent
  derivations agreeing.

  I also re-fetched the five upstream sources at their pinned SHAs into docs/models/gemma4-upstream.txt; the eight line citations I locked all resolve.

  Two corrections worth flagging

  - gemma-4.md:776 claimed the sandbox cannot reach huggingface.co. It can — that false claim is why the upstream investigation looked unreproducible.
  - <bos> does not appear in Google's canonical template; it emits {{- bos_token -}}. My first assertion failed on this and it's a genuine distinction, now recorded in
  chat_template.json.

  Mutation-tested

  Seven mutations against gemma-4.md and both transcripts each fired exactly one failure; d8/d9/d10 each demonstrated failing against the pre-fix scripts. During M5 I deleted
  a transcript line and recovered it exactly by diffing against the E4B copy — file restored to 6348 lines, templates byte-identical.

  Two things needing you

  The committed transcripts are now stale against the fixed generators. That's recorded as a strict xfail, so it fails loudly once regenerated — that's the signal to delete
  the marker:

  ./scripts/models/gguf_header_transcript.sh --out docs/models/gemma4-gguf.txt \
      /mnt/ai-data/models/gguf/gemma-4-E2B-it/*.gguf \
      /mnt/ai-data/models/gguf/gemma-4-E4B-it/*.gguf
  ./scripts/models/gemma4_header_transcript.sh --out docs/models/gemma4-safetensors.txt \
      /mnt/ai-data/models/hf/gemma-4-E2B /mnt/ai-data/models/hf/gemma-4-E4B
  make test

  Two pre-existing failures in tests/training/test_promote_scorecard.py — both from git rev-parse HEAD failing, which is the linked-worktree limitation, not my changes. make 
  quality passes regardless; worth confirming they're green on the workstation.

  For Phase 2, the 2304 width is now pinned correctly in the contract and the ~15 divergent sites are listed in readers — they'll be corrected under the guard rather than
  before it.

✻ Crunched for 22m 16s

❯ Great work! I ran the commands you provided and we have some bugs to fix before phase 2:                                                                                     
                                                                                                                                                                               
  _________________________________________________________ test_transcript_line_citations_resolve[gemma-4.md:758->614]                                                        
  _________________________________________________________                                                                                                                    
                                                                                                                                                                               
  citing_doc = 'docs/models/gemma-4.md', citing_line = 758, cited_file = 'docs/models/gemma4-gguf.txt', cited_line = 614, needle = 'blk.{L}.attn_k.weight [1536, 256]'         
                                                                                                                                                                               
  >   ???                                                                                                                                                                      
  E   AssertionError: docs/models/gemma-4.md:758 cites docs/models/gemma4-gguf.txt line 614 for 'blk.{L}.attn_k.weight [1536, 256]', but that line now reads: '     28         
  blk.{L}.attn_output.weight [2048, 1536] BF16'                                                                                                                                
  E   assert 'blk.{L}.attn_k.weight [1536, 256]' in '     28  blk.{L}.attn_output.weight [2048, 1536] BF16'                                                                    
                                                                                                                                                                               
  /work/tests/models/test_gemma4_facts_contract.py:528: AssertionError                                                                                                         
  _________________________________________________________ test_transcript_line_citations_resolve[gemma-4.md:758->626]                                                        
  _________________________________________________________                                                                                                                    
                                                                                                                                                                               
  citing_doc = 'docs/models/gemma-4.md', citing_line = 758, cited_file = 'docs/models/gemma4-gguf.txt', cited_line = 626, needle = 'blk.{L}.attn_k.weight [1536, 512]'         
                                                                                                                                                                               
  >   ???                                                                                                                                                                      
  E   AssertionError: docs/models/gemma-4.md:758 cites docs/models/gemma4-gguf.txt line 626 for 'blk.{L}.attn_k.weight [1536, 512]', but that line now reads: '      7         

  - gemma-4.md:776 claimed the sandbox cannot reach huggingface.co. It can — that false claim is why the upstream investigation looked unreproducible.
  - <bos> does not appear in Google's canonical template; it emits {{- bos_token -}}. My first assertion failed on this and it's a genuine distinction, now recorded in
  chat_template.json.

  Mutation-tested

  Seven mutations against gemma-4.md and both transcripts each fired exactly one failure; d8/d9/d10 each demonstrated failing against the pre-fix scripts. During M5 I deleted
  a transcript line and recovered it exactly by diffing against the E4B copy — file restored to 6348 lines, templates byte-identical.

  Two things needing you

  The committed transcripts are now stale against the fixed generators. That's recorded as a strict xfail, so it fails loudly once regenerated — that's the signal to delete
  the marker:

  ./scripts/models/gguf_header_transcript.sh --out docs/models/gemma4-gguf.txt \
      /mnt/ai-data/models/gguf/gemma-4-E2B-it/*.gguf \
      /mnt/ai-data/models/gguf/gemma-4-E4B-it/*.gguf
  ./scripts/models/gemma4_header_transcript.sh --out docs/models/gemma4-safetensors.txt \
      /mnt/ai-data/models/hf/gemma-4-E2B /mnt/ai-data/models/hf/gemma-4-E4B
  make test

  Two pre-existing failures in tests/training/test_promote_scorecard.py — both from git rev-parse HEAD failing, which is the linked-worktree limitation, not my changes. make
  quality passes regardless; worth confirming they're green on the workstation.

  For Phase 2, the 2304 width is now pinned correctly in the contract and the ~15 divergent sites are listed in readers — they'll be corrected under the guard rather than
  before it.