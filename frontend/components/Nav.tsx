"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

/**
 * Navigation is the product only.
 *
 * Evaluation and observability are engineering concerns: they run in the
 * backend, are driven from the CLI (`scripts/run_eval.py`, `scripts/ablate.py`)
 * and the REST API, and are covered by the test suite. They were taking up
 * two of five top-level slots and competing with the thing users actually came
 * to do, so they are no longer in the interface at all.
 *
 * The routes still exist and still work - nothing was deleted - they are simply
 * not advertised. Re-add an entry here to bring one back.
 */
const PRIMARY = [
  { href: "/", label: "Research" },
  { href: "/documents", label: "Documents" },
  { href: "/extract", label: "Extract" },
  { href: "/architecture", label: "How it works" },
];

export function Nav() {
  const pathname = usePathname();
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const stored = (() => {
      try {
        return localStorage.getItem("legallens.theme") as "light" | "dark" | null;
      } catch {
        return null;
      }
    })();
    const resolved =
      stored ?? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    setTheme(resolved);
    document.documentElement.setAttribute("data-theme", resolved);
  }, []);

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("legallens.theme", next);
    } catch {
      /* non-fatal */
    }
  }

  return (
    <header
      className="sticky top-0 z-40 border-b backdrop-blur-md"
      style={{ borderColor: "var(--border)", background: "color-mix(in srgb, var(--bg) 88%, transparent)" }}
    >
      <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-6 px-5">
        <Link href="/" className="flex items-center gap-2.5 shrink-0">
          <span
            className="grid h-7 w-7 place-items-center rounded-[7px] text-[13px] font-bold text-white"
            style={{ background: "linear-gradient(140deg, var(--color-accent-500), var(--color-accent-700))" }}
          >
            L
          </span>
          <span className="text-[15px] font-semibold tracking-tight">LegalLens</span>
        </Link>

        <nav className="flex items-center gap-0.5 overflow-x-auto">
          {PRIMARY.map((link) => {
            const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className="rounded-lg px-3 py-1.5 text-[13.5px] font-medium whitespace-nowrap transition-colors"
                style={
                  active
                    ? { background: "var(--surface-sunken)", color: "var(--text)" }
                    : { color: "var(--text-muted)" }
                }
              >
                {link.label}
              </Link>
            );
          })}

        </nav>

        <div className="ml-auto flex items-center gap-3">
          <span
            className="hidden text-[11.5px] sm:block"
            style={{ color: "var(--text-muted)" }}
            title="LegalLens retrieves and summarises what documents say. It does not give legal advice."
          >
            Research tool · not legal advice
          </span>
          <button
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            className="grid h-8 w-8 place-items-center rounded-lg border text-[13px] transition-colors"
            style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </div>
      </div>
    </header>
  );
}
