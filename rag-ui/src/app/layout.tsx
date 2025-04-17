import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'RAG Demo',
  description: 'A demo of Retrieval Augmented Generation using Next.js and FastAPI',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <main className="container mx-auto px-4 py-8 max-w-4xl">
          {children}
        </main>
      </body>
    </html>
  );
} 