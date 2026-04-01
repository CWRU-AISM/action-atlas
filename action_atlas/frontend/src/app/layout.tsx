import type { Metadata } from "next";
import "./globals.css";
import { ReduxProvider } from "@/redux/provider";

export const metadata: Metadata = {
  title: "VLA Feature Explorer",
  description: "Visual Analytics for Vision-Language-Action Model Interpretability",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap"
        />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Limelight&display=swap"
        />
      </head>
      <body
        className={`antialiased font-roboto`}
        style={{ fontFamily: "Roboto, sans-serif" }}
      >
        <ReduxProvider>{children}</ReduxProvider>
      </body>
    </html>
  );
}
