"use client";

import { ReactNode, useState } from "react";

export function Card({ children, className = "", ...rest }: { children: ReactNode; className?: string } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...rest}
      className={`rounded-xl border ${className}`}
      style={{ borderColor: "var(--border)", background: "var(--surface)", ...(rest.style ?? {}) }}
    >
      {children}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad" | "accent";
  title?: string;
}) {
  const tones: Record<string, { bg: string; fg: string }> = {
    neutral: { bg: "var(--surface-sunken)", fg: "var(--text-muted)" },
    good: { bg: "var(--color-good-100)", fg: "var(--color-good-700)" },
    warn: { bg: "var(--color-warn-100)", fg: "var(--color-warn-700)" },
    bad: { bg: "var(--color-bad-100)", fg: "var(--color-bad-700)" },
    accent: { bg: "var(--color-accent-100)", fg: "var(--color-accent-700)" },
  };
  const t = tones[tone];
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium whitespace-nowrap"
      style={{ background: t.bg, color: t.fg }}
    >
      {children}
    </span>
  );
}

export function Disclosure({
  label,
  children,
  defaultOpen = false,
  right,
}: {
  label: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  right?: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 rounded-md py-1 text-left text-[12px] font-medium transition-colors"
        style={{ color: "var(--text-muted)" }}
        aria-expanded={open}
      >
        <span
          className="inline-block transition-transform duration-150"
          style={{ transform: open ? "rotate(90deg)" : "none" }}
        >
          ›
        </span>
        {label}
        {right && <span className="ml-auto">{right}</span>}
      </button>
      {open && <div className="pt-1.5">{children}</div>}
    </div>
  );
}

export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden className="animate-spin">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.2" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: string; tone?: "good" | "warn" | "bad" }) {
  const color =
    tone === "good" ? "var(--color-good-500)" : tone === "warn" ? "var(--color-warn-500)" : tone === "bad" ? "var(--color-bad-500)" : "var(--text)";
  return (
    <Card className="p-3.5">
      <div className="text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="mt-1 text-[22px] font-semibold tabular-nums leading-tight" style={{ color }}>
        {value}
      </div>
      {hint && (
        <div className="mt-0.5 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
          {hint}
        </div>
      )}
    </Card>
  );
}

export function EmptyState({ title, body, action }: { title: string; body: string; action?: ReactNode }) {
  return (
    <div className="grid place-items-center px-6 py-16 text-center">
      <div className="max-w-md">
        <h3 className="text-[15px] font-semibold">{title}</h3>
        <p className="mt-1.5 text-[13.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
          {body}
        </p>
        {action && <div className="mt-4">{action}</div>}
      </div>
    </div>
  );
}

export function Button({
  children,
  variant = "secondary",
  ...rest
}: { children: ReactNode; variant?: "primary" | "secondary" | "ghost" } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const styles: Record<string, React.CSSProperties> = {
    primary: { background: "var(--color-accent-600)", color: "#fff", borderColor: "transparent" },
    secondary: { background: "var(--surface)", color: "var(--text)", borderColor: "var(--border)" },
    ghost: { background: "transparent", color: "var(--text-muted)", borderColor: "transparent" },
  };
  return (
    <button
      {...rest}
      className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[13px] font-medium transition-opacity disabled:cursor-not-allowed disabled:opacity-50 ${rest.className ?? ""}`}
      style={{ ...styles[variant], ...(rest.style ?? {}) }}
    >
      {children}
    </button>
  );
}
