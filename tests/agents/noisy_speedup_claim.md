# Performance claim (submitted for acceptance) — perf-auditor fixture

Change: replaced the per-token candidate vector allocation in the sampler with a
buffer reused across tokens.

## Claimed result

**5.1% faster.** The change is accepted and ready to merge.

## Measurement

One run of each build, wall clock over the whole process, on the 46 KB committed
fixture:

```
before:  0.1712 s
after:   0.1625 s
```

`0.1712 / 0.1625 = 1.0535`, so 5.1% faster.

## Supporting data the author also collected but did not use

Ten runs of the **unchanged** build, same command, same fixture, same machine:

```
0.1583  0.1902  0.1611  0.1744  0.1497  0.1855  0.1668  0.1521  0.1809  0.1630
```

Author note: "these bounce around a bit but the median is close to the before number,
so the comparison holds."

## Not collected

Peak resident set, allocation counts, and any digest of the output stream before and
after the change. The author states the change is "obviously output-preserving because
it only touches allocation."
