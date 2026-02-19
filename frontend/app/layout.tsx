import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "DeepRead",
  description: "ML Paper Comprehension Agent",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-100 text-slate-900">{children}</body>
    </html>
  );
}

