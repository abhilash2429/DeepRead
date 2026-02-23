"use client";

import Link from "next/link";
import styles from "./page.module.css";

const flowSvgMarkup = String.raw`<svg class="${styles.flowSvg}" viewBox="0 0 1060 640" fill="none" xmlns="http://www.w3.org/2000/svg" style="font-family: 'DM Mono', monospace;">
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#7c6af7" opacity="0.8"/>
    </marker>
    <marker id="arrow2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#4fd1a5" opacity="0.8"/>
    </marker>
    <marker id="arrowgray" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#666" opacity="0.8"/>
    </marker>
    <filter id="sketch">
      <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="2" result="noise"/>
      <feDisplacementMap in="SourceGraphic" in2="noise" scale="1.2" xChannelSelector="R" yChannelSelector="G"/>
    </filter>
  </defs>

  <g filter="url(#sketch)">
    <rect x="40" y="40" width="150" height="64" rx="8" fill="rgba(124,106,247,0.08)" stroke="#7c6af7" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="115" y="66" text-anchor="middle" fill="#7c6af7" font-size="10" letter-spacing="1">PDF UPLOAD</text>
  <text x="115" y="82" text-anchor="middle" fill="#c8c8e0" font-size="12" font-weight="600">User Paper</text>
  <text x="115" y="96" text-anchor="middle" fill="rgba(200,200,224,0.4)" font-size="9">or arXiv link</text>

  <path d="M 192 72 L 248 72" stroke="#7c6af7" stroke-width="1.5" stroke-dasharray="3,2" marker-end="url(#arrow)" opacity="0.7"/>

  <g filter="url(#sketch)">
    <rect x="250" y="40" width="160" height="64" rx="8" fill="rgba(255,255,255,0.02)" stroke="rgba(200,200,224,0.25)" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="330" y="63" text-anchor="middle" fill="rgba(200,200,224,0.6)" font-size="10" letter-spacing="1">SERVICE</text>
  <text x="330" y="80" text-anchor="middle" fill="#c8c8e0" font-size="12" font-weight="600">PDF Parser</text>
  <text x="330" y="95" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="9">pymupdf  fitz</text>

  <path d="M 412 72 L 468 72" stroke="rgba(200,200,224,0.3)" stroke-width="1.5" stroke-dasharray="3,2" marker-end="url(#arrowgray)" opacity="0.7"/>

  <g filter="url(#sketch)">
    <rect x="470" y="40" width="160" height="64" rx="8" fill="rgba(255,255,255,0.02)" stroke="rgba(200,200,224,0.25)" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="550" y="63" text-anchor="middle" fill="rgba(200,200,224,0.6)" font-size="10" letter-spacing="1">SERVICE</text>
  <text x="550" y="80" text-anchor="middle" fill="#c8c8e0" font-size="12" font-weight="600">Vision Service</text>
  <text x="550" y="95" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="9">gemini-2.5-flash</text>

  <path d="M 632 72 L 688 72" stroke="rgba(200,200,224,0.3)" stroke-width="1.5" stroke-dasharray="3,2" marker-end="url(#arrowgray)" opacity="0.7"/>

  <g filter="url(#sketch)">
    <rect x="690" y="28" width="180" height="88" rx="8" fill="rgba(124,106,247,0.06)" stroke="#7c6af7" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="780" y="52" text-anchor="middle" fill="#7c6af7" font-size="10" letter-spacing="1">AGENT 1</text>
  <text x="780" y="72" text-anchor="middle" fill="#c8c8e0" font-size="13" font-weight="700">Ingestion</text>
  <text x="780" y="88" text-anchor="middle" fill="rgba(200,200,224,0.4)" font-size="9">ParsedPaper  DB</text>
  <text x="780" y="103" text-anchor="middle" fill="rgba(200,200,224,0.3)" font-size="8">figures + text + equations</text>

  <text x="780" y="130" text-anchor="middle" fill="rgba(124,106,247,0.6)" font-size="9"> ParsedPaper</text>

  <path d="M 780 118 L 780 168" stroke="#7c6af7" stroke-width="1.5" stroke-dasharray="3,2" marker-end="url(#arrow)" opacity="0.7"/>

  <g filter="url(#sketch)">
    <rect x="620" y="170" width="320" height="88" rx="8" fill="rgba(124,106,247,0.1)" stroke="#7c6af7" stroke-width="2" stroke-dasharray="5,3"/>
  </g>
  <text x="780" y="193" text-anchor="middle" fill="#7c6af7" font-size="10" letter-spacing="1">AGENT 2  MOST CRITICAL</text>
  <text x="780" y="215" text-anchor="middle" fill="#e8e8f0" font-size="14" font-weight="700">Comprehension Agent</text>
  <text x="780" y="232" text-anchor="middle" fill="rgba(200,200,224,0.45)" font-size="9">Full paper text  InternalRepresentation</text>
  <text x="780" y="247" text-anchor="middle" fill="rgba(200,200,224,0.3)" font-size="8">gemini-2.5-pro  1M context  one call  cached to DB</text>

  <text x="620" y="276" fill="rgba(200,200,224,0.4)" font-size="8" text-anchor="middle">component_graph</text>
  <text x="720" y="276" fill="rgba(247,106,106,0.6)" font-size="8" text-anchor="middle">ambiguity_log</text>
  <text x="820" y="276" fill="rgba(247,209,106,0.6)" font-size="8" text-anchor="middle">hyperparameters</text>
  <text x="940" y="276" fill="rgba(200,200,224,0.4)" font-size="8" text-anchor="middle">prerequisites</text>

  <path d="M 780 260 L 780 308" stroke="#7c6af7" stroke-width="1.5" stroke-dasharray="3,2" marker-end="url(#arrow)" opacity="0.7"/>
  <text x="795" y="288" fill="rgba(124,106,247,0.6)" font-size="8">InternalRepresentation</text>

  <text x="530" y="332" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="9" letter-spacing="2">LANGGRAPH PIPELINE — 6 NODES IN SEQUENCE</text>

  <g filter="url(#sketch)">
    <rect x="40" y="345" width="140" height="72" rx="6" fill="rgba(79,209,165,0.06)" stroke="#4fd1a5" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="110" y="365" text-anchor="middle" fill="#4fd1a5" font-size="9" letter-spacing="0.5">SECTION 1</text>
  <text x="110" y="382" text-anchor="middle" fill="#c8c8e0" font-size="11" font-weight="600">What It Does</text>
  <text x="110" y="397" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="8">plain English</text>
  <text x="110" y="409" text-anchor="middle" fill="rgba(79,209,165,0.4)" font-size="8">flash</text>

  <path d="M 182 381 L 210 381" stroke="rgba(79,209,165,0.4)" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arrow2)" opacity="0.6"/>

  <g filter="url(#sketch)">
    <rect x="212" y="345" width="140" height="72" rx="6" fill="rgba(124,106,247,0.08)" stroke="#7c6af7" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="282" y="365" text-anchor="middle" fill="#7c6af7" font-size="9" letter-spacing="0.5">SECTION 2</text>
  <text x="282" y="382" text-anchor="middle" fill="#c8c8e0" font-size="11" font-weight="600">The Mechanism</text>
  <text x="282" y="397" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="8">equations decoded</text>
  <text x="282" y="409" text-anchor="middle" fill="rgba(124,106,247,0.5)" font-size="8">pro </text>

  <path d="M 354 381 L 382 381" stroke="rgba(79,209,165,0.4)" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arrow2)" opacity="0.6"/>

  <g filter="url(#sketch)">
    <rect x="384" y="345" width="140" height="72" rx="6" fill="rgba(79,209,165,0.06)" stroke="#4fd1a5" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="454" y="365" text-anchor="middle" fill="#4fd1a5" font-size="9" letter-spacing="0.5">SECTION 3</text>
  <text x="454" y="382" text-anchor="middle" fill="#c8c8e0" font-size="11" font-weight="600">Prerequisites</text>
  <text x="454" y="397" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="8">dependency order</text>
  <text x="454" y="409" text-anchor="middle" fill="rgba(79,209,165,0.4)" font-size="8">flash</text>

  <path d="M 526 381 L 554 381" stroke="rgba(79,209,165,0.4)" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arrow2)" opacity="0.6"/>

  <g filter="url(#sketch)">
    <rect x="556" y="345" width="140" height="72" rx="6" fill="rgba(124,106,247,0.08)" stroke="#7c6af7" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="626" y="365" text-anchor="middle" fill="#7c6af7" font-size="9" letter-spacing="0.5">SECTION 4</text>
  <text x="626" y="382" text-anchor="middle" fill="#c8c8e0" font-size="11" font-weight="600">Impl. Map</text>
  <text x="626" y="397" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="8">labeled code</text>
  <text x="626" y="409" text-anchor="middle" fill="rgba(124,106,247,0.5)" font-size="8">pro </text>

  <path d="M 698 381 L 726 381" stroke="rgba(79,209,165,0.4)" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arrow2)" opacity="0.6"/>

  <g filter="url(#sketch)">
    <rect x="728" y="345" width="140" height="72" rx="6" fill="rgba(247,106,106,0.06)" stroke="#f76a6a" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="798" y="365" text-anchor="middle" fill="#f76a6a" font-size="9" letter-spacing="0.5">SECTION 5</text>
  <text x="798" y="382" text-anchor="middle" fill="#c8c8e0" font-size="11" font-weight="600">Left Out</text>
  <text x="798" y="397" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="8">ambiguity report</text>
  <text x="798" y="409" text-anchor="middle" fill="rgba(124,106,247,0.5)" font-size="8">pro </text>

  <path d="M 870 381 L 898 381" stroke="rgba(79,209,165,0.4)" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arrow2)" opacity="0.6"/>

  <g filter="url(#sketch)">
    <rect x="900" y="345" width="140" height="72" rx="6" fill="rgba(79,209,165,0.06)" stroke="#4fd1a5" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="970" y="365" text-anchor="middle" fill="#4fd1a5" font-size="9" letter-spacing="0.5">SECTION 6</text>
  <text x="970" y="382" text-anchor="middle" fill="#c8c8e0" font-size="11" font-weight="600">Training Recipe</text>
  <text x="970" y="397" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="8">hyperparams table</text>
  <text x="970" y="409" text-anchor="middle" fill="rgba(79,209,165,0.4)" font-size="8">flash</text>

  <path d="M 530 419 L 530 460" stroke="rgba(79,209,165,0.4)" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arrow2)" opacity="0.6"/>

  <text x="545" y="445" fill="rgba(200,200,224,0.3)" font-size="8"> DB write per section</text>

  <g filter="url(#sketch)">
    <rect x="40" y="462" width="920" height="38" rx="6" fill="rgba(124,106,247,0.04)" stroke="rgba(124,106,247,0.2)" stroke-width="1" stroke-dasharray="6,3"/>
  </g>
  <text x="500" y="479" text-anchor="middle" fill="rgba(124,106,247,0.5)" font-size="9" letter-spacing="1">SSE STREAM </text>
  <text x="500" y="492" text-anchor="middle" fill="rgba(200,200,224,0.3)" font-size="8">thinking events + section_token events  ThinkingStream.tsx renders live</text>

  <path d="M 280 502 L 280 540" stroke="rgba(200,200,224,0.2)" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arrowgray)" opacity="0.5"/>
  <path d="M 530 502 L 530 540" stroke="rgba(200,200,224,0.2)" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arrowgray)" opacity="0.5"/>
  <path d="M 780 502 L 780 540" stroke="rgba(200,200,224,0.2)" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#arrowgray)" opacity="0.5"/>

  <g filter="url(#sketch)">
    <rect x="40" y="542" width="200" height="72" rx="6" fill="rgba(247,162,106,0.06)" stroke="#f7a26a" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="140" y="562" text-anchor="middle" fill="#f7a26a" font-size="9" letter-spacing="0.5">AGENT 3  Q&amp;A</text>
  <text x="140" y="580" text-anchor="middle" fill="#c8c8e0" font-size="12" font-weight="600">QA Agent</text>
  <text x="140" y="596" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="8">gemini-2.5-flash  tools</text>
  <text x="140" y="608" text-anchor="middle" fill="rgba(200,200,224,0.25)" font-size="7">ConversationSummaryBufferMemory</text>

  <g filter="url(#sketch)">
    <rect x="300" y="542" width="200" height="72" rx="6" fill="rgba(79,209,165,0.06)" stroke="#4fd1a5" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="400" y="562" text-anchor="middle" fill="#4fd1a5" font-size="9" letter-spacing="0.5">OUTPUT</text>
  <text x="400" y="580" text-anchor="middle" fill="#c8c8e0" font-size="12" font-weight="600">Briefing Document</text>
  <text x="400" y="596" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="8">6 sections  streamed live</text>
  <text x="400" y="609" text-anchor="middle" fill="rgba(200,200,224,0.25)" font-size="7">renders as user reads</text>

  <g filter="url(#sketch)">
    <rect x="560" y="542" width="200" height="72" rx="6" fill="rgba(124,106,247,0.06)" stroke="#7c6af7" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="660" y="562" text-anchor="middle" fill="#7c6af7" font-size="9" letter-spacing="0.5">AGENT 4  CODE</text>
  <text x="660" y="580" text-anchor="middle" fill="#c8c8e0" font-size="12" font-weight="600">Code Agent</text>
  <text x="660" y="596" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="8">gemini-2.5-flash</text>
  <text x="660" y="609" text-anchor="middle" fill="rgba(200,200,224,0.25)" font-size="7">PyTorch  ASSUMED/INFERRED labels</text>

  <g filter="url(#sketch)">
    <rect x="820" y="542" width="200" height="72" rx="6" fill="rgba(247,209,106,0.06)" stroke="#f7d16a" stroke-width="1.5" stroke-dasharray="4,3"/>
  </g>
  <text x="920" y="562" text-anchor="middle" fill="#f7d16a" font-size="9" letter-spacing="0.5">OUTPUT</text>
  <text x="920" y="580" text-anchor="middle" fill="#c8c8e0" font-size="12" font-weight="600">Artifacts</text>
  <text x="920" y="596" text-anchor="middle" fill="rgba(200,200,224,0.35)" font-size="8">.md  .py  .csv</text>
  <text x="920" y="609" text-anchor="middle" fill="rgba(200,200,224,0.25)" font-size="7">downloadable on demand</text>

  <rect x="40" y="630" width="10" height="10" rx="2" fill="rgba(124,106,247,0.3)" stroke="#7c6af7" stroke-width="1"/>
  <text x="56" y="639" fill="rgba(200,200,224,0.4)" font-size="8">gemini-2.5-pro (deep reasoning)</text>
  <rect x="240" y="630" width="10" height="10" rx="2" fill="rgba(79,209,165,0.2)" stroke="#4fd1a5" stroke-width="1"/>
  <text x="256" y="639" fill="rgba(200,200,224,0.4)" font-size="8">gemini-2.5-flash (speed)</text>
  <rect x="420" y="630" width="10" height="10" rx="2" fill="rgba(247,106,106,0.2)" stroke="#f76a6a" stroke-width="1"/>
  <text x="436" y="639" fill="rgba(200,200,224,0.4)" font-size="8">ambiguity / missing</text>
  <rect x="580" y="630" width="10" height="10" rx="2" fill="rgba(247,209,106,0.2)" stroke="#f7d16a" stroke-width="1"/>
  <text x="596" y="639" fill="rgba(200,200,224,0.4)" font-size="8">artifact output</text>
</svg>`;

export default function HomePage() {
  return (
    <main className={styles.page}>
      <div className={styles.noiseOverlay} />
      <div className={styles.gridBg} />

      <nav className={styles.nav}>
        <div className={styles.logo}>
          <div className={styles.logoDot} />
          DeepRead
        </div>
        <Link href="/upload" className={styles.navCta}>
          Get Started - It&apos;s Free
        </Link>
      </nav>

      <div className={styles.hero}>
        <div className={styles.heroGlow} />
        <div className={styles.heroTag}>Agentic ML Paper Comprehension</div>
        <h1 className={styles.heroH1}>
          Papers are written
          <br />
          for <em>reviewers.</em>
        </h1>
        <h2 className={styles.heroSub}>
          We rewrite them
          <br />
          for builders.
        </h2>
        <p className={styles.heroDesc}>
          Drop any ML research paper. DeepRead reads it the way an expert would - decoding every equation,
          flagging every ambiguity, generating labeled implementation code - then hands you the brief your
          advisor never wrote.
        </p>
        <div className={styles.heroCtaRow}>
          <Link href="/upload" className={styles.btnPrimary}>
            Analyze a Paper
          </Link>
          <a href="#how-it-works" className={styles.btnGhost}>
            See how it works
          </a>
        </div>
      </div>

      <hr className={styles.divider} />

      <section className={styles.section}>
        <div className={styles.sectionLabel}>The Problem</div>
        <h2 className={styles.sectionH2}>
          The gap no one
          <br />
          talks about
        </h2>
        <p className={styles.sectionDesc}>
          Every ML paper has two versions: the one that gets published, and the one you need to implement it.
          DeepRead bridges the distance between them.
        </p>

        <div className={styles.gapSection}>
          <div className={`${styles.gapCol} ${styles.before}`}>
            <div className={styles.gapColLabel}>
              <div className={styles.gapDot} /> What the paper gives you
            </div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>-</span><span>Equations with undefined symbols - W<sub>q</sub>, d<sub>k</sub>, t - no explanation of what they are</span></div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>-</span><span>Hyperparameters buried in footnotes, appendices, or omitted entirely</span></div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>-</span><span>&quot;We use standard initialization&quot; - which one? Xavier? Kaiming? They don&apos;t say.</span></div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>-</span><span>Architecture diagrams with no implementation consequence explained</span></div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>-</span><span>Citations to 5 other papers you now also have to read</span></div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>-</span><span>Training details split across the paper, appendix, and Table 3 footnote</span></div>
          </div>
          <div className={`${styles.gapCol} ${styles.after}`}>
            <div className={styles.gapColLabel}>
              <div className={styles.gapDot} /> What DeepRead gives you
            </div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>+</span><span>Every symbol decoded at the point of use - never left undefined</span></div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>+</span><span>Full hyperparameter table - paper-stated, inferred, or missing with agent default</span></div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>+</span><span>Every assumption labeled explicitly: <span className={`${styles.badge} ${styles.badgeYellow}`}>ASSUMED</span> with reason and consequence</span></div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>+</span><span>Figures interpreted by vision model - components, arrows, dimensions described</span></div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>+</span><span>Prerequisite concepts explained inline - problem  solution  paper-specific usage</span></div>
            <div className={styles.gapItem}><span className={styles.gapItemIcon}>+</span><span>Training recipe synthesized from all paper sections into one clean document</span></div>
          </div>
        </div>
      </section>

      <hr className={styles.divider} />

      <section id="how-it-works" className={styles.section}>
        <div className={styles.sectionLabel}>How It Works</div>
        <h2 className={styles.sectionH2}>
          The <em>agentic</em> pipeline
        </h2>
        <p className={styles.sectionDesc}>
          Seven specialized agents work in sequence. Each one has a single job. The output of each feeds the next.
        </p>

        <div className={styles.flowWrap}>
          <div className={styles.flowTitle}>- DeepRead AGENTIC FLOW -</div>
          <div dangerouslySetInnerHTML={{ __html: flowSvgMarkup }} />
        </div>
      </section>

      <hr className={styles.divider} />

      <section className={styles.section}>
        <div className={styles.sectionLabel}>What You Get</div>
        <h2 className={styles.sectionH2}>
          Six sections.
          <br />
          <em>Everything</em> you need.
        </h2>
        <p className={styles.sectionDesc}>
          The briefing is not a summary. It is the paper transformed - same information density, reorganized for implementation.
        </p>

        <div className={styles.outputGrid}>
          <div className={styles.outputCard} style={{ ["--card-accent" as string]: "#4fd1a5" }}>
            <div className={styles.outputCardNum}>01 -</div>
            <div className={styles.outputCardTitle}>What This Paper Actually Does</div>
            <div className={styles.outputCardDesc}>One paragraph. No jargon. No prior ML knowledge assumed. Written for the version of you that hasn&apos;t read the paper yet.</div>
            <div><span className={`${styles.badge} ${styles.badgeGreen}`}>plain english</span></div>
          </div>

          <div className={styles.outputCard} style={{ ["--card-accent" as string]: "#7c6af7" }}>
            <div className={styles.outputCardNum}>02 -</div>
            <div className={styles.outputCardTitle}>The Mechanism</div>
            <div className={styles.outputCardDesc}>Every equation decoded inline. Every symbol defined at point of use. Every figure interpreted. Prerequisite concepts explained before they appear.</div>
            <div><span className={`${styles.badge} ${styles.badgeGreen}`}>paper-stated</span> <span className={`${styles.badge} ${styles.badgeYellow}`}>inferred</span></div>
          </div>

          <div className={styles.outputCard} style={{ ["--card-accent" as string]: "#4fd1a5" }}>
            <div className={styles.outputCardNum}>03 -</div>
            <div className={styles.outputCardTitle}>What You Need To Already Know</div>
            <div className={styles.outputCardDesc}>Prerequisites in dependency order. Each one: the problem it solved, what it does, why this paper uses it specifically.</div>
            <div><span className={`${styles.badge} ${styles.badgeGreen}`}>dependency ordered</span></div>
          </div>

          <div className={styles.outputCard} style={{ ["--card-accent" as string]: "#7c6af7" }}>
            <div className={styles.outputCardNum}>04 -</div>
            <div className={styles.outputCardTitle}>The Full Implementation Map</div>
            <div className={styles.outputCardDesc}>Every component in build order. PyTorch snippets with inline equation citations. Every assumption labeled. Every inference explained.</div>
            <div><span className={`${styles.badge} ${styles.badgeGreen}`}>paper-stated</span> <span className={`${styles.badge} ${styles.badgeYellow}`}>ASSUMED</span> <span className={`${styles.badge} ${styles.badgeRed}`}>missing</span></div>
          </div>

          <div className={styles.outputCard} style={{ ["--card-accent" as string]: "#f76a6a" }}>
            <div className={styles.outputCardNum}>05 -</div>
            <div className={styles.outputCardTitle}>What The Paper Left Out</div>
            <div className={styles.outputCardDesc}>Every ambiguity surfaced. Every missing hyperparameter flagged. Implementation consequence for every unresolved decision. You can override each one.</div>
            <div><span className={`${styles.badge} ${styles.badgeRed}`}>ambiguity report</span></div>
          </div>

          <div className={styles.outputCard} style={{ ["--card-accent" as string]: "#f7d16a" }}>
            <div className={styles.outputCardNum}>06 -</div>
            <div className={styles.outputCardTitle}>How To Train It</div>
            <div className={styles.outputCardDesc}>Full training recipe synthesized from every section, footnote, and appendix. Hyperparameter table with source and status for every value.</div>
            <div><span className={`${styles.badge} ${styles.badgeYellow}`}>hyperparams</span> <span className={`${styles.badge} ${styles.badgeGreen}`}>training recipe</span></div>
          </div>
        </div>
      </section>

      <hr className={styles.divider} />

      <section className={styles.section}>
        <div className={styles.sectionLabel}>Live Agent Activity</div>
        <h2 className={styles.sectionH2}>
          Watch it
          <br />
          <em>think.</em>
        </h2>
        <p className={styles.sectionDesc}>
          No progress bars. No fake loading states. A live stream of exactly what the agent is doing at the moment it&apos;s doing it.
        </p>

        <div className={styles.thinkingPreview}>
          <div className={styles.thinkingLine}><span className={styles.thinkingArrow}>{">"}</span><span style={{ color: "rgba(200,200,224,0.9)" }}>Reading abstract and identifying core contribution...</span></div>
          <div className={styles.thinkingLine}><span className={styles.thinkingArrow}>{">"}</span><span style={{ color: "rgba(200,200,224,0.7)" }}>Found Algorithm 1 block on page 6 - extracting pseudocode...</span></div>
          <div className={styles.thinkingLine}><span className={styles.thinkingArrow}>{">"}</span><span style={{ color: "rgba(200,200,224,0.5)" }}>Interpreting Figure 2 - encoder-decoder attention diagram...</span></div>
          <div className={styles.thinkingLine}><span className={styles.thinkingArrow}>{">"}</span><span style={{ color: "rgba(200,200,224,0.3)" }}>Found 6 undefined hyperparameters across appendix B and Table 3 footnote...</span></div>
          <div className={styles.cursorWrap}><span className={styles.cursorBlink} /></div>
        </div>
      </section>

      <hr className={styles.divider} />

      <section className={`${styles.section} ${styles.finalCtaSection}`}>
        <div className={styles.sectionLabel}>Ready</div>
        <h2 className={styles.sectionH2}>
          Stop reading papers.
          <br />
          Start <em>implementing</em> them.
        </h2>
        <p className={styles.sectionDesc}>Free for your first 3 papers. No credit card.</p>
        <Link href="/upload" className={styles.btnPrimary}>
          Analyze Your First Paper
        </Link>
      </section>

      <footer className={styles.footer}>
        <div className={styles.logo}>
          <div className={styles.logoDot} />
          DeepRead
        </div>
        <span>Built for engineers who implement, not just cite.</span>
        <span className={styles.footerMeta}>gemini-2.5-pro  langgraph  prisma</span>
      </footer>
    </main>
  );
}

