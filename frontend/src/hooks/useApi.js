import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Loads data from an API call, optionally polling. `refresh()` re-runs immediately.
 * Polling pauses while the tab is hidden.
 */
export function useApi(fetcher, deps = [], { pollMs = 0, enabled = true } = {}) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(enabled);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const result = await fetcherRef.current();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return undefined;
    load(false);
    if (!pollMs) return undefined;
    const id = setInterval(() => {
      if (document.visibilityState === "visible") load(true);
    }, pollMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, pollMs, enabled]);

  return { data, error, loading, refresh: () => load(true), setData };
}
