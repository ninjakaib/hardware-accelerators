import Header from "./components/Header";
import Hero from "./components/Hero";
import InfoPanel from "./components/InfoPanel";
import Footer from "./components/Footer";

export default function Page() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Header />
      <main>
        <Hero />
        <InfoPanel />
      </main>
      <Footer />
    </div>
  );
}
