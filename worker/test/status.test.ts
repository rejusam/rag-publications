import { describe, expect, it } from "vitest";
import { splitStatus, NONE_FALLBACK } from "../src/status.ts";

describe("splitStatus", () => {
  it.each([
    ["STATUS: answered\nRodents spread it.", "answered", "Rodents spread it."],
    ["status: PARTIAL\n\nOnly the first part.", "partial", "Only the first part."],
    ["\n  STATUS: none  \nNot covered.", "none", "Not covered."],
    ["**STATUS: none**\nNot covered.", "none", "Not covered."],
    ["`STATUS: answered`.\nYes.", "answered", "Yes."],
  ])("reads %j", (input, status, text) => {
    expect(splitStatus(input)).toEqual({ status, text });
  });
  it("defaults to answered and keeps text when the line is missing", () => {
    expect(splitStatus("Rodents spread it.")).toEqual({ status: "answered", text: "Rodents spread it." });
  });
  it("ignores a STATUS line that is not first", () => {
    const t = "Rodents spread it.\nSTATUS: none";
    expect(splitStatus(t)).toEqual({ status: "answered", text: t });
  });
  it("ignores unknown values", () => {
    expect(splitStatus("STATUS: maybe\nX")).toEqual({ status: "answered", text: "STATUS: maybe\nX" });
  });
  it("fills an empty none answer with an honest sentence", () => {
    expect(splitStatus("STATUS: none\n")).toEqual({ status: "none", text: NONE_FALLBACK });
  });
  it("leaves an empty answered answer empty (pipeline raises EmptyAnswerError)", () => {
    expect(splitStatus("STATUS: answered")).toEqual({ status: "answered", text: "" });
  });
});

describe("no-answer wording", () => {
  it("falls back to the plain line the cards use", () => {
    expect(NONE_FALLBACK).toBe("These papers don’t cover that.");
  });
});
