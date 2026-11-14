"use client";

import { useState } from "react";
import Image from "next/image";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setIsLoading(true);
    setAnswer("");

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setAnswer(data.answer || "No answer returned from API.");
    } catch (error) {
      setAnswer(`Error: ${error instanceof Error ? error.message : "Unknown error occurred"}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black p-4">
      <main className="flex w-full max-w-3xl flex-col items-center gap-8 py-12 px-6 bg-white dark:bg-zinc-900 rounded-2xl shadow-sm border border-zinc-200 dark:border-zinc-800">
        <div className="flex flex-col items-center gap-2 text-center">
          <Image
            className="dark:invert h-8 w-auto"
            src="/next.svg"
            alt="Next.js logo"
            width={120}
            height={24}
            priority
          />
          <h1 className="text-3xl font-bold tracking-tight text-black dark:text-zinc-50">
            Clause AI Assistant
          </h1>
          <p className="text-zinc-600 dark:text-zinc-400">
            Ask a question to interact with the FastAPI backend
          </p>
        </div>

        <form onSubmit={handleAsk} className="w-full flex flex-col gap-4">
          <div className="relative">
            <textarea
              className="w-full p-4 rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-black dark:text-white focus:ring-2 focus:ring-blue-500 outline-none transition-all resize-none"
              placeholder="Enter your question here..."
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>
          <button
            type="submit"
            disabled={isLoading || !question.trim()}
            className="flex h-12 w-full items-center justify-center gap-2 rounded-full bg-black text-white transition-colors hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Thinking...
              </span>
            ) : (
              "Ask Question"
            )}
          </button>
        </form>

        {answer && (
          <div className="w-full p-6 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 animate-in fade-in slide-in-from-bottom-2 duration-300">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-2">
              Answer
            </h2>
            <p className="text-black dark:text-zinc-200 whitespace-pre-wrap leading-relaxed">
              {answer}
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
