import { useEffect, useRef, useState } from "react";
import { fetchPlayers } from "../api";

interface Props {
  onSearch: (player: string) => void;
  loading?: boolean;
}

export default function SearchBar({ onSearch, loading }: Props) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (query.length < 2) {
      setSuggestions([]);
      return;
    }
    debounceRef.current = setTimeout(() => {
      fetchPlayers(query)
        .then(setSuggestions)
        .catch(() => setSuggestions([]));
    }, 200);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query]);

  const submit = (name: string) => {
    setQuery(name);
    setOpen(false);
    onSearch(name);
  };

  return (
    <div className="relative w-full max-w-xl">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (query.trim()) submit(query.trim());
        }}
      >
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            placeholder="Search a player — e.g. Bellingham"
            className="flex-1 rounded-xl border border-pitch-700 bg-pitch-800 px-4 py-3 text-white placeholder:text-pitch-300/50 focus:border-pitch-500 focus:outline-none focus:ring-1 focus:ring-pitch-500"
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="rounded-xl bg-pitch-500 px-6 py-3 font-medium text-white transition hover:bg-pitch-300 disabled:opacity-50"
          >
            {loading ? "..." : "Find"}
          </button>
        </div>
      </form>

      {open && suggestions.length > 0 && (
        <ul className="absolute z-20 mt-2 w-full overflow-hidden rounded-xl border border-pitch-700 bg-pitch-800 shadow-xl">
          {suggestions.map((name) => (
            <li key={name}>
              <button
                type="button"
                onClick={() => submit(name)}
                className="w-full px-4 py-2.5 text-left text-sm hover:bg-pitch-700"
              >
                {name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
