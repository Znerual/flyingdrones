"""Strict checkpoint loading.

Several upstream evaluation scripts load weights with ``strict=False``; a key-name
mismatch then silently leaves (part of) the network at random initialisation. All
benchmark evaluators load through ``load_state_dict_checked``, which fails loudly
unless every model key is matched.
"""


def load_state_dict_checked(model, state_dict, strip_prefixes=("module.",), allow_missing=(), name="model"):
    """Load weights and raise when keys do not match.

    ``strip_prefixes`` are removed when *all* checkpoint keys carry them (e.g. the
    ``module.`` prefix of ``DataParallel``). ``allow_missing`` lists key prefixes that
    may legitimately stay at initialisation.
    """
    sd = dict(state_dict)
    for pre in strip_prefixes:
        if sd and all(k.startswith(pre) for k in sd):
            sd = {k[len(pre):]: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if not any(k.startswith(a) for a in allow_missing)]
    unexpected = list(res.unexpected_keys)
    n_model = len(model.state_dict())
    print(f"[{name}] loaded {n_model - len(res.missing_keys)}/{n_model} keys; "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError(f"[{name}] weight/key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return res
