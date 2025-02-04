"use client";

import { useEffect, useState } from "react";

export default function Hero() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <section className="relative h-screen flex items-center justify-center overflow-hidden">
      <div
        className={`absolute inset-0 flex flex-col items-center justify-center transition-opacity duration-1000 ${
          isVisible ? "opacity-100" : "opacity-0"
        }`}
      >
        <p className="text-xl md:text-2xl mb-4 text-gray-600">
          the l-mul algorithm
        </p>
        <h1 className="text-4xl md:text-6xl font-bold text-center leading-tight">
          floating point multiplication,
          <br />
          <span className="text-5xl md:text-7xl bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600 animate-glow">
            faster
          </span>
        </h1>
      </div>
    </section>
  );
}
