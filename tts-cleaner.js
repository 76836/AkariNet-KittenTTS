(() => {
  const EMOJI_DATA_CDN =
    "https://cdn.jsdelivr.net/npm/unicode-emoji-json@0.4.0/data-by-emoji.json";

  async function loadEmojiData() {
    const res = await fetch(EMOJI_DATA_CDN);
    if (!res.ok) throw new Error(`Failed to load emoji data: ${res.status}`);
    return res.json();
  }

  function cleanMarkdownForTTS(text, emojiData = {}, options = {}) {
    const { sentencesPerParagraph = 4 } = options;
    if (!text || !text.trim()) return "";

    let clean = text;

    // 1. Tables
    clean = clean.replace(/^[|].*[|]$/gm, "");
    clean = clean.replace(/^[-|: ]+$/gm, "");

    // 2. List / blockquote markers
    clean = clean.replace(/^\s*\d+\.\s+/gm, "");
    clean = clean.replace(/^\s*[-*+>—•]\s*/gm, "");

    // 3. Inline Markdown
    clean = clean.replace(/^#{1,6}\s*/gm, "");
    clean = clean.replace(/(\*{1,3}|_{1,3})(.*?)\1/gs, "$2");
    clean = clean.replace(/```[\s\S]*?```/g, "");
    clean = clean.replace(/`([^`]+)`/g, "$1");
    clean = clean.replace(/~~(.*?)~~/g, "$1");
    clean = clean.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
    clean = clean.replace(/https?:\/\/\S+/g, "");
    clean = clean.replace(/[#_~`]/g, "");
    clean = clean.replace(/<[^>]+>/g, "");
    clean = clean.replace(/^[-*_]{3,}$/gm, "");

    // 4. Emoji replacement
    const emojiRegex =
      /(\p{Extended_Pictographic}(\u200D\p{Extended_Pictographic})*\uFE0F?|\p{Emoji_Presentation}|\d\uFE0F\u20E3|[\u2700-\u27BF]|\u00A9|\u00AE)/gu;

    if (typeof Intl !== "undefined" && Intl.Segmenter) {
      const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
      let result = "";
      for (const { segment } of segmenter.segment(clean)) {
        emojiRegex.lastIndex = 0;
        if (emojiRegex.test(segment)) {
          const entry = emojiData[segment];
          result += entry ? ` ${entry.name} ` : " ";
        } else {
          result += segment;
        }
      }
      clean = result;
    } else {
      clean = clean.replace(emojiRegex, (match) => {
        const entry = emojiData[match];
        return entry ? ` ${entry.name} ` : " ";
      });
    }

    // 5. Whitespace normalisation
    clean = clean.replace(/[ \t]{2,}/g, " ");
    clean = clean.split("\n").map(l => l.trim()).filter(Boolean).join(" ").trim();
    if (!clean) return "";

    // 6. Group into paragraphs
    const sentenceRe = /[^.!?]*[.!?]+["']?/g;
    const matches = clean.match(sentenceRe) || [];
    const tail = clean.replace(sentenceRe, "").trim();
    const sentences = [
      ...matches.map(s => s.trim()).filter(Boolean),
      ...(tail ? [`${tail}.`] : []),
    ];
    const paragraphs = [];
    for (let i = 0; i < sentences.length; i += sentencesPerParagraph) {
      paragraphs.push(sentences.slice(i, i + sentencesPerParagraph).join(" "));
    }
    return paragraphs.join("\n\n") || "";
  }

  window.TTSCleaner = { cleanMarkdownForTTS, loadEmojiData };
})();
