import { describe, expect, it } from "vitest";
import {
  filterCitations,
  normaliseCitations,
  stripMarkdown,
  stripReasoning,
} from "../src/postprocess.ts";
import type { Source } from "../src/types.ts";

const src = (title: string, authors_short: string, year: number): Source => ({
  title, authors_short, year, journal: null, doi: null,
});

const FILTER_SOURCES: Source[] = [
  src("Travel time paper", "John et al.", 2024),
  src("Land use paper", "Rulli et al.", 2025),
  src("Ebola paper", "Hayman et al.", 2022),
];

describe("stripReasoning", () => {
  it.each([
    ["Plain answer.", "Plain answer."],
    ["<think>secret steps</think>Final.", "Final."],
    ["<THINK>\nmulti\nline\n</THINK>\n  Final.  ", "Final."],
    ["Answer. <think>trailing unterminated", "Answer."],
    ["<think>only reasoning", ""],
  ])("%j", (raw, expected) => {
    expect(stripReasoning(raw)).toBe(expected);
  });
});

describe("normaliseCitations", () => {
  it.each([
    ["Rodents spread it 【John et al., 2024】.", "Rodents spread it (John et al., 2024)."],
    ["A 【Smith et al., 2020】 and B 【Jones et al., 2021】.", "A (Smith et al., 2020) and B (Jones et al., 2021)."],
    ["No brackets here.", "No brackets here."],
    ["Stray open 【 only.", "Stray open ( only."],
    ["Stray close 】 only.", "Stray close ) only."],
  ])("%j", (raw, expected) => {
    expect(normaliseCitations(raw)).toBe(expected);
  });
});

describe("stripMarkdown", () => {
  it.each([
    ["**Inhalation** of dust", "Inhalation of dust"],
    ["shed by *M. natalensis* rodents", "shed by M. natalensis rodents"],
    ["- first\n- second", "first\nsecond"],
    ["## Heading\ntext", "Heading\ntext"],
    ["2 * 3 = 6", "2 * 3 = 6"],
    ["Plain text.", "Plain text."],
    ["- 0.5 correlation", "- 0.5 correlation"],
    ["- 3 cases were reported", "- 3 cases were reported"],
    ["#1 risk factor", "#1 risk factor"],
    ["-  0.5 corr", "-  0.5 corr"],
    ["-\t2 x", "-\t2 x"],
    ["*  bold item", "bold item"],
  ])("%j", (raw, expected) => {
    expect(stripMarkdown(raw)).toBe(expected);
  });
});

describe("filterCitations", () => {
  it.each([
    ["Rodents shed virus (Safronetz et al., 2022; Wozniak et al., 2021).", "Rodents shed virus."],
    ["Contact spreads it (Rulli et al., 2025).", "Contact spreads it (Rulli et al., 2025)."],
    ["Mixed (Rulli et al., 2025; Lo Iacono et al., 2015).", "Mixed (Rulli et al., 2025)."],
    ["(John et al., 2024 — Modelling Lassa virus dynamics in West)", "(John et al., 2024 — Modelling Lassa virus dynamics in West)"],
    ["during the (2014–2016 outbreak)", "during the (2014–2016 outbreak)"],
    ["No parentheses here.", "No parentheses here."],
  ])("basic %j", (raw, expected) => {
    expect(filterCitations(raw, FILTER_SOURCES)).toBe(expected);
  });

  it.each([
    ["(Smith et al., 2019 – title)", ""],
    ["(Smith et al., 2019, p. 4)", ""],
    ["(Smith et al.,2019)", ""],
    ["(Smith et al., 2019, 2020)", ""],
    ["((Smith et al., 2019))", ""],
    ["(see Rulli et al., 2025)", "(see Rulli et al., 2025)"],
    ["(e.g. Rulli et al., 2025)", "(e.g. Rulli et al., 2025)"],
    ["(in 2019)", "(in 2019)"],
    ["(March 2020)", "(March 2020)"],
    ["(since 2018)", "(since 2018)"],
    ["(Mastomys natalensis, 2019)", "(Mastomys natalensis, 2019)"],
    ["(R0 ≈ 1.5; 2019)", "(R0 ≈ 1.5; 2019)"],
    ["Values fell to .5", "Values fell to .5"],
    ["Wait ... really , yes.", "Wait ... really , yes."],
    ["Rats matter. (Lo Iacono et al., 2015) showed that rats matter. End.", "Rats matter. Showed that rats matter. End."],
    ["(Smith et al., 2019; 2020)", ""],
    ["(Hayman et al., 2022)", "(Hayman et al., 2022)"],
    ["(Hayman, 2022)", ""],
    ["A\n(Smith et al., 2019).", "A\n."],
  ])("strict grammar %j", (raw, expected) => {
    expect(filterCitations(raw, FILTER_SOURCES)).toBe(expected);
  });

  it.each([
    ["First. (Smith et al., 2019) showed R0 was 1.5 in Sierra Leone. Second sentence stays.", "First. Showed R0 was 1.5 in Sierra Leone. Second sentence stays."],
    ['First. (Smith, 2019) found "rats." Next one. Another. Last.', 'First. Found "rats." Next one. Another. Last.'],
    ["(Smith, 2019) showed it (John et al., 2024). Keep.", "Showed it (John et al., 2024). Keep."],
    ["First. (Smith et al., 2019) found rats\n\nNew paragraph here. Stays.", "First. Found rats\n\nNew paragraph here. Stays."],
    ["First. (Smith et al., 2019) showed Dr. Smith was right. Next.", "First. Showed Dr. Smith was right. Next."],
    ["A.\r\n(Smith et al., 2019) found rats.\r\nNext.", "A.\r\nFound rats.\r\nNext."],
    ["(a;b 2019)", "(a;b 2019)"],
  ])("capitalises sentence start %j", (raw, expected) => {
    expect(filterCitations(raw, FILTER_SOURCES)).toBe(expected);
  });

  it.each([
    ["shed virus (Fichet‑Calvet et al., 2014).", "shed virus."],
    ["shed virus (Fichet‐Calvet et al., 2014).", "shed virus."],
    ["(Fichet-Calvet et al., 2014)", ""],
  ])("drops unicode-hyphen surname %j", (raw, expected) => {
    expect(filterCitations(raw, FILTER_SOURCES)).toBe(expected);
  });

  it("keeps unicode-hyphen surname matching an ASCII source", () => {
    const sources = [...FILTER_SOURCES, src("Fichet-Calvet paper", "Fichet-Calvet et al.", 2014)];
    const raw = "shed virus (Fichet‑Calvet et al., 2014).";
    expect(filterCitations(raw, sources)).toBe(raw);
  });

  it.each([
    ["shed (Müller et al., 2019).", "shed."],
    ["shed (Gómez et al., 2020).", "shed."],
    ["shed (Ødegaard et al., 2018).", "shed."],
    ["shed (van der Berg et al., 2017).", "shed."],
    ["shed (in 2019).", "shed (in 2019)."],
    ["shed (de facto, 2019).", "shed (de facto, 2019)."],
  ])("unicode and particle surnames %j", (raw, expected) => {
    expect(filterCitations(raw, FILTER_SOURCES)).toBe(expected);
  });

  it.each([
    ["Gómez et al.", 2020, "shed (Gómez et al., 2020)."],
    ["van der Berg et al.", 2017, "shed (van der Berg et al., 2017)."],
  ])("keeps in-corpus %s", (authors, year, raw) => {
    const sources = [...FILTER_SOURCES, src("Extra paper", authors, year)];
    expect(filterCitations(raw, sources)).toBe(raw);
  });

  it("decomposed umlaut still parses as a citation", () => {
    expect(filterCitations("shed (Müller et al., 2019).", FILTER_SOURCES)).toBe("shed.");
  });

  it("long whitespace run is fast", () => {
    const raw = "(a" + " ".repeat(20000) + "x 2019)";
    const start = performance.now();
    filterCitations(raw, FILTER_SOURCES);
    expect(performance.now() - start).toBeLessThan(200);
  });

  it("many dropped sentence starts is fast", () => {
    const raw = "Sentence. (Smith et al., 2019) rats spread it. ".repeat(5000);
    const start = performance.now();
    filterCitations(raw, FILTER_SOURCES);
    expect(performance.now() - start).toBeLessThan(300);
  });

  it("a realistic 500-word answer filters in under 2 ms (Workers CPU budget)", () => {
    const sentence =
      "Lassa virus spreads through rodent contact (John et al., 2024; Lo Iacono et al., 2015) and human movement. ";
    const raw = sentence.repeat(30); // about 500 words, 60 citation groups
    filterCitations(raw, FILTER_SOURCES); // warm the regex cache
    const start = performance.now();
    filterCitations(raw, FILTER_SOURCES);
    expect(performance.now() - start).toBeLessThan(2);
  });

  it("chained unknown citations drop without eating spaces", () => {
    // Regression test: multiple back-to-back unknown citations with zero space
    // between groups was losing the trailing spaces from before the chain.
    // scanSoFarBackwards must walk back through empty out pieces, not just
    // check the last element.
    const raw = "X  (Smith et al., 2019)(Jones et al., 2020)(Wilson et al., 2021) end.";
    expect(filterCitations(raw, FILTER_SOURCES)).toBe("X  end.");
  });

  it("surname with curly apostrophe is kept when in corpus", () => {
    // Surnames like O'Brien (with curly apostrophe U+2019, a look-alike
    // substitution gpt-oss is known to make) must be recognised and matched
    // against retrieved papers.
    const sources = [...FILTER_SOURCES, src("Test paper", "O’Brien et al.", 2019)];
    const raw = "shed virus (O’Brien et al., 2019).";
    expect(filterCitations(raw, sources)).toBe(raw);
  });

  it("surname with curly apostrophe is dropped when not in corpus", () => {
    // When a curly-apostrophe surname is not in the retrieved sources, it
    // should be dropped like any other unknown citation.
    const raw = "shed virus (O’Brien et al., 2019).";
    expect(filterCitations(raw, FILTER_SOURCES)).toBe("shed virus.");
  });
});
