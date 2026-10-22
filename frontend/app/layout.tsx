import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PhD Finder — Find your next research home",
  description:
    "Explore PhD research leads grounded in official university sources, matched to your interests.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
