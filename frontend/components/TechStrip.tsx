"use client";

import Link from "next/link";

/**
 * A one-line "what powers this screen" strip under each page title.
 *
 * The point is that the machinery is inspectable from anywhere in the product,
 * not buried in a repo. Each chip names a real technique or library; the strip
 * links through to the Architecture page where each one is explained in full.
 */
export function TechStrip({ items, className = "" }: { items: string[]; className?: string }) {
  return (
    <div className={`flex flex-wrap items-center gap-1.5 ${className}`}>
      <span className="text-[10.5px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
        Powered by
      </span>
      {items.map((item) => (
        <span
          key={item}
          className="rounded-md px-1.5 py-0.5 font-mono text-[10.5px]"
          style={{ background: "var(--surface-sunken)", color: "var(--text-muted)" }}
        >
          {item}
        </span>
      ))}
      <Link
        href="/architecture"
        className="text-[10.5px] underline underline-offset-2"
        style={{ color: "var(--text-muted)" }}
      >
        how it works →
      </Link>
    </div>
  );
}
