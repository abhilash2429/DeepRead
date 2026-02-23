import type { Metadata } from "next";
import { DM_Mono, Instrument_Serif, Syne } from "next/font/google";

import "./globals.css";
import { Providers } from "./Providers";

const syne = Syne({ subsets: ["latin"], variable: "--font-sans" });
const dmMono = DM_Mono({ subsets: ["latin"], variable: "--font-mono", weight: ["400", "500"] });
const instrumentSerif = Instrument_Serif({ subsets: ["latin"], variable: "--font-serif", weight: "400" });

export const metadata: Metadata = {
  title: "DeepRead",
  description: "Drop a research paper. Get everything you need to implement it.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${syne.variable} ${dmMono.variable} ${instrumentSerif.variable}`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}

