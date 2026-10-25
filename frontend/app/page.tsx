"use client";

import { FormEvent, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Source = {
  university: string;
  country: string;
  url: string;
  score: number;
  excerpt: string;
  fetched_at: string;
};

type Result = {
  answer: string;
  sources: Source[];
  source_count: number;
  chat_model: string;
};

type Tab = "ask" | "match";

const countries = [
  "Australia",
  "Canada",
  "China",
  "France",
  "Germany",
  "India",
  "Japan",
  "Netherlands",
  "United Kingdom",
  "United States",
];

const suggestions = [
  "Which UK universities have AI PhD research opportunities?",
  "What should I know about PhD funding in Canada?",
  "Find doctoral programmes in computational biology",
];

// The model cites sources as [S3]; turn each into a link to source card 3.
// Also accept the variants it sometimes writes: 【S3】 and [S3] with a zero-width space.
function linkCitations(answer: string): string {
  return answer.replace(/[\[【]\u200b?\s*S(\d+)\s*[\]】]/g, "[S$1](#source-$1)");
}

// The prompt asks for Markdown only, but models sometimes put HTML lists inside table
// cells (Markdown cells can't hold lists). react-markdown shows such tags as literal
// text, so turn the common ones into inline bullets instead.
function flattenListTags(answer: string): string {
  return answer
    .replace(/<\/?[uo]l\s*>/gi, "")
    .replace(/<li\s*>/gi, " • ")
    .replace(/<\/li\s*>/gi, "")
    .replace(/<br\s*\/?>/gi, " ");
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("ask");
  const [question, setQuestion] = useState("");
  const [selectedCountries, setSelectedCountries] = useState<string[]>([]);
  const [field, setField] = useState("");
  const [degree, setDegree] = useState("");
  const [interests, setInterests] = useState("");
  const [funding, setFunding] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [apiReady, setApiReady] = useState<boolean | null>(null);

  useEffect(() => {
    fetch("/api/backend/api/v1/health", { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error("API unavailable");
        const body = await response.json();
        setApiReady(Boolean(body.ready));
      })
      .catch(() => setApiReady(false));
  }, []);

  function toggleCountry(country: string) {
    setSelectedCountries((current) =>
      current.includes(country)
        ? current.filter((item) => item !== country)
        : [...current, country],
    );
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const path = tab === "ask" ? "ask" : "match";
    const payload =
      tab === "ask"
        ? { question, countries: selectedCountries, top_k: 12 }
        : {
            field,
            degree,
            research_interests: interests
              .split(",")
              .map((item) => item.trim())
              .filter(Boolean),
            preferred_countries: selectedCountries,
            ...(funding.trim() ? { budget_or_funding_need: funding.trim() } : {}),
            top_k: 12,
          };

    try {
      const response = await fetch(`/api/backend/api/v1/${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(typeof body.detail === "string" ? body.detail : "The request could not be completed.");
      }
      setResult(body as Result);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="site-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="PhD Finder home">
          <span className="brand-mark">P</span>
          <span>fieldnotes<span className="brand-period">.</span></span>
        </a>
        <div className="topbar-right">
          <span className={`status-dot ${apiReady ? "is-ready" : ""}`} />
          <span className="status-copy">
            {apiReady === null ? "Checking research index" : apiReady ? "Research index ready" : "Index needs setup"}
          </span>
          <a className="nav-link" href="#how-it-works">How it works <span aria-hidden="true">↗</span></a>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-line" /> A clearer path to your PhD</p>
          <h1>Find the research<br />that <em>moves you.</em></h1>
          <p className="hero-description">
            Explore doctoral research leads from official university sources, shaped around your curiosity and goals.
          </p>
          <div className="hero-proof">
            <div className="avatar-stack" aria-hidden="true"><span>R</span><span>M</span><span>A</span></div>
            <span>Grounded in official university pages</span>
          </div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="orbit orbit-one" /><div className="orbit orbit-two" />
          <div className="sun-disc"><span>FIELD<br />NOTES</span></div>
          <span className="art-label label-top">curiosity, meet opportunity</span>
          <span className="art-label label-bottom">10 countries · one starting point</span>
          <span className="spark spark-one">✳</span><span className="spark spark-two">✳</span>
        </div>
      </section>

      <section className="workspace" aria-label="Find PhD research leads">
        <div className="workspace-heading">
          <div><p className="section-kicker">YOUR NEXT CHAPTER</p><h2>Let’s find your fit.</h2></div>
          <span className="step-count"><span>01</span> / 02</span>
        </div>

        <div className="search-card">
          <div className="tab-bar" role="tablist" aria-label="Search method">
            <button className={`tab ${tab === "ask" ? "active" : ""}`} role="tab" aria-selected={tab === "ask"} onClick={() => { setTab("ask"); setError(""); setResult(null); }} type="button">
              <span className="tab-icon">✳</span> Ask a question
            </button>
            <button className={`tab ${tab === "match" ? "active" : ""}`} role="tab" aria-selected={tab === "match"} onClick={() => { setTab("match"); setError(""); setResult(null); }} type="button">
              <span className="tab-icon">◎</span> Match my profile
            </button>
          </div>

          <form onSubmit={submit} className="search-form">
            {tab === "ask" ? (
              <>
                <label className="field-label" htmlFor="question">What would you like to discover?</label>
                <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="e.g. Which universities in Germany offer funded PhDs in climate science?" minLength={3} maxLength={2000} required rows={3} />
                <div className="suggestions"><span className="suggestion-label">TRY ASKING</span>{suggestions.map((suggestion) => <button key={suggestion} type="button" className="suggestion-chip" onClick={() => setQuestion(suggestion)}>{suggestion}</button>)}</div>
              </>
            ) : (
              <>
                <p className="field-label">A little about your research journey</p>
                <div className="form-grid">
                  <label className="input-wrap">Field of study <span className="required">*</span><input value={field} onChange={(event) => setField(event.target.value)} placeholder="e.g. Computer science" required /></label>
                  <label className="input-wrap">Your degree & preparation <span className="required">*</span><input value={degree} onChange={(event) => setDegree(event.target.value)} placeholder="e.g. MSc, 3.8 GPA, thesis in NLP" required /></label>
                  <label className="input-wrap full-width">Research interests <span className="required">*</span><input value={interests} onChange={(event) => setInterests(event.target.value)} placeholder="Separate interests with commas" required /></label>
                  <label className="input-wrap full-width">Funding needs <span className="optional">OPTIONAL</span><input value={funding} onChange={(event) => setFunding(event.target.value)} placeholder="e.g. Need a fully funded position" /></label>
                </div>
              </>
            )}

            <div className="country-section">
              <div className="country-heading"><span className="field-label">Where would you like to study?</span><span className="optional">OPTIONAL · SELECT ANY</span></div>
              <div className="country-list">
                {countries.map((country) => (
                  <button key={country} type="button" className={`country-chip ${selectedCountries.includes(country) ? "selected" : ""}`} onClick={() => toggleCountry(country)} aria-pressed={selectedCountries.includes(country)}>
                    {selectedCountries.includes(country) && <span aria-hidden="true">✓ </span>}{country}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-footer">
              <span className="privacy-note"><span aria-hidden="true">↗</span> Your search is matched to official sources</span>
              <button className="submit-button" type="submit" disabled={loading || apiReady === false}>
                {loading ? "Searching…" : tab === "ask" ? "Explore research leads" : "Find my matches"}
                <span aria-hidden="true">→</span>
              </button>
            </div>
            {apiReady === false && <p className="inline-note">The backend is not ready yet. Start the API and build its research index to search.</p>}
          </form>
        </div>
      </section>

      {(error || result) && <section className="results-section" aria-live="polite">
        {error && <div className="error-box"><strong>We couldn’t complete that search.</strong><p>{error}</p></div>}
        {result && <>
          <div className="results-heading"><div><p className="section-kicker">YOUR RESEARCH STARTING POINTS</p><h2>Here’s what we found.</h2></div><span className="result-count">{result.source_count} {result.source_count === 1 ? "source" : "sources"}</span></div>
          <article className="answer-card"><div className="answer-label"><span>✳</span> A GROUNDED OVERVIEW</div><div className="answer-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{linkCitations(flattenListTags(result.answer))}</ReactMarkdown></div><div className="answer-meta">Generated with {result.chat_model} · Research leads, not admissions decisions</div></article>
          <div className="source-grid">{result.sources.map((source, index) => <article className="source-card" id={`source-${index + 1}`} key={`${source.url}-${index}`}>
            <div className="source-topline"><span className="source-number">S{index + 1}</span><span className="source-country">{source.country}</span><span className="source-score">{Math.round(source.score * 100)}% match</span></div>
            <h3>{source.university}</h3><p className="source-excerpt">“{source.excerpt}”</p>
            <div className="source-footer"><span>Official source</span><a href={source.url} target="_blank" rel="noreferrer">Visit page <span aria-hidden="true">↗</span></a></div>
          </article>)}</div>
          <p className="verification-note">Use these as starting points, and verify current requirements on each official university page.</p>
        </>}
      </section>}

      <section className="how-section" id="how-it-works"><div className="how-intro"><p className="section-kicker">BUILT FOR THE BIG QUESTION</p><h2>Less searching.<br /><em>More imagining.</em></h2></div><div className="how-steps"><article><span className="how-number">01</span><h3>Start with what matters</h3><p>Tell us what you want to study, where you want to go, or what you’re still figuring out.</p></article><article><span className="how-number">02</span><h3>Follow the evidence</h3><p>We search curated official university pages and bring relevant passages back to you.</p></article><article><span className="how-number">03</span><h3>Make it your own</h3><p>Explore the sources, check the details, and take the next step with confidence.</p></article></div></section>

      <footer className="footer"><a className="brand footer-brand" href="#top"><span className="brand-mark">P</span><span>fieldnotes<span className="brand-period">.</span></span></a><p>Thoughtful research leads, backed by official sources.</p><span>© 2026 Fieldnotes · Not an admissions decision service</span></footer>
    </main>
  );
}
