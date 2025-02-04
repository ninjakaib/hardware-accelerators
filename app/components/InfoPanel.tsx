"use client";
import ReactMarkdown from "react-markdown";
import { useEffect, useState } from "react";
import Image from "next/image";

export default function InfoPanel() {
  const [content, setContent] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadContent() {
      try {
        setIsLoading(true);
        const response = await fetch(
          "/hardware-accelerators-site/content/technical-report.md",
          {
            headers: {
              "Content-Type": "text/plain",
              "Cache-Control": "no-cache",
            },
          }
        );

        if (!response.ok) {
          throw new Error(
            `Failed to load content: ${response.status} ${response.statusText}`
          );
        }

        const text = await response.text();
        setContent(text);
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : "An error occurred while loading the content";
        setError(errorMessage);
        console.error("Error loading markdown:", error);
      } finally {
        setIsLoading(false);
      }
    }

    loadContent();
  }, []);

  return (
    <section className="bg-white py-16">
      <div className="container mx-auto px-4 max-w-3xl">
        {isLoading && (
          <div className="text-center py-8">
            <p>Loading content...</p>
          </div>
        )}

        {error && (
          <div className="text-red-600 text-center py-8">
            <p>Error loading content: {error}</p>
          </div>
        )}

        {!isLoading && !error && (
          <article className="prose prose-lg prose-gray mx-auto">
            <ReactMarkdown
              components={{
                img: ({ src, alt }) => (
                  <div className="relative w-full h-64">
                    <Image
                      src={src || ""}
                      alt={alt || ""}
                      fill
                      className="rounded-lg object-cover"
                    />
                  </div>
                ),
                a: ({ children, href }) => (
                  <a href={href} className="text-blue-600 hover:text-blue-800">
                    {children}
                  </a>
                ),
                h1: ({ children }) => (
                  <h1 className="text-4xl font-bold mb-4 text-black">
                    {children}
                  </h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-3xl font-bold mt-8 mb-4 text-black">
                    {children}
                  </h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-2xl font-bold mt-6 mb-3 text-black">
                    {children}
                  </h3>
                ),
                h4: ({ children }) => (
                  <h4 className="text-2xl font-bold mt-6 mb-3 text-black">
                    {children}
                  </h4>
                ),
                p: ({ children }) => (
                  <p className="mb-4 text-gray-800 leading-relaxed">
                    {children}
                  </p>
                ),
                ul: ({ children }) => (
                  <ul className="list-disc pl-6 mb-4 space-y-2">{children}</ul>
                ),
                li: ({ children }) => (
                  <li className="text-gray-800">{children}</li>
                ),
              }}
            >
              {content}
            </ReactMarkdown>
          </article>
        )}
      </div>
    </section>
  );
}
