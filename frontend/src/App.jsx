import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import html2canvas from 'html2canvas'
import './App.css'

const VIS_STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'if', 'to', 'of', 'in', 'on', 'for', 'with',
  'from', 'by', 'at', 'is', 'are', 'was', 'were', 'be', 'been', 'it', 'that', 'this',
  'as', 'i', 'you', 'he', 'she', 'they', 'we', 'my', 'your', 'our', 'their', 'me',
  'him', 'her', 'them', 'so', 'not', 'too', 'very', 'just', 'what', 'when', 'where',
  'why', 'how', 'who', 'whom', 'which', 'then', 'than', 'also', 'there', 'here',
])

const EMO_HINT_WORDS = new Set([
  'pain', 'suffer', 'suffering', 'grief', 'mourning', 'death', 'kill', 'cry', 'alone',
  'love', 'fear', 'rage', 'dark', 'loss', 'magic', 'god', 'family', 'alive', 'dead',
])

const IDEA_CLUSTERS = [
  ['song', 'music', 'dance', 'dancing', 'rhythm'],
  ['death', 'dying', 'dead', 'mourning', 'grief', 'loss'],
  ['family', 'tree', 'historic', 'ancestors', 'relatives'],
  ['magic', 'god', 'sacred', 'soul', 'spirit'],
  ['world', 'build', 'built', 'story', 'create', 'creation'],
  ['want', 'desire', 'wish', 'longing', 'crave'],
]

function splitSentences(rawText) {
  if (!rawText?.trim()) return []
  return rawText
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)
}

function normalizeWord(word) {
  const cleaned = (word || '').toLowerCase().replace(/[^a-z']/g, '')
  if (!cleaned) return ''
  if (cleaned === "'s" || cleaned === 's' || cleaned === "'" || cleaned === "s'") return ''
  if (/^'[a-z]$/.test(cleaned)) return ''
  return cleaned
}

function stemWord(word) {
  const token = normalizeWord(word)
  if (!token) return ''
  const suffixes = ['ingly', 'edly', 'ing', 'ed', 'ly', 'es', 's']
  for (const suffix of suffixes) {
    if (token.endsWith(suffix) && token.length > suffix.length + 2) {
      return token.slice(0, -suffix.length)
    }
  }
  return token
}

function soundex(word) {
  const token = normalizeWord(word)
  if (!token) return ''
  const map = {
    b: '1', f: '1', p: '1', v: '1', c: '2', g: '2', j: '2', k: '2', q: '2',
    s: '2', x: '2', z: '2', d: '3', t: '3', l: '4', m: '5', n: '5', r: '6',
  }
  const first = token[0].toUpperCase()
  let prev = map[token[0]] || ''
  const out = []
  for (const ch of token.slice(1)) {
    const code = map[ch] || ''
    if (code && code !== prev) out.push(code)
    prev = code
  }
  return (first + out.join('') + '000').slice(0, 4)
}

function vowelGroupCount(word) {
  const token = normalizeWord(word)
  if (!token) return 0
  const matches = token.match(/[aeiouy]+/g)
  return matches ? matches.length : 0
}

function bigramDiceSimilarity(left, right) {
  const a = normalizeWord(left)
  const b = normalizeWord(right)
  if (a.length < 2 || b.length < 2) return 0

  const gramsA = []
  const gramsB = []
  for (let i = 0; i < a.length - 1; i += 1) gramsA.push(a.slice(i, i + 2))
  for (let i = 0; i < b.length - 1; i += 1) gramsB.push(b.slice(i, i + 2))

  const counts = {}
  gramsA.forEach((gram) => {
    counts[gram] = (counts[gram] || 0) + 1
  })

  let overlap = 0
  gramsB.forEach((gram) => {
    if (counts[gram]) {
      overlap += 1
      counts[gram] -= 1
    }
  })

  return (2 * overlap) / (gramsA.length + gramsB.length)
}

function longestCommonSubsequenceLength(left, right) {
  const a = normalizeWord(left)
  const b = normalizeWord(right)
  const rows = a.length + 1
  const cols = b.length + 1
  if (!a || !b) return 0

  const dp = Array.from({ length: rows }, () => Array(cols).fill(0))
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }
  return dp[a.length][b.length]
}

function isIdeaRelated(left, right) {
  const a = normalizeWord(left)
  const b = normalizeWord(right)
  if (!a || !b) return false
  return IDEA_CLUSTERS.some((cluster) => cluster.includes(a) && cluster.includes(b) && a !== b)
}

function stripSilentEndingE(word) {
  const token = normalizeWord(word)
  if (!token) return ''
  if (token.length > 2 && token.endsWith('e') && !token.endsWith('le') && !token.endsWith('ye')) {
    return token.slice(0, -1)
  }
  return token
}

function smallWordHalfOverlapConnects(left, right) {
  const a = stripSilentEndingE(left)
  const b = stripSilentEndingE(right)
  if (!a || !b || a === b) return false

  const small = a.length <= b.length ? a : b
  const large = a.length <= b.length ? b : a
  if (small.length < 2 || small.length > 3) return false

  const smallLetters = [...new Set(small.split(''))]
  const largeLetters = new Set(large.split(''))
  const sharedCount = smallLetters.filter((ch) => largeLetters.has(ch)).length
  const sharedRatio = sharedCount / smallLetters.length
  if (sharedRatio < 0.5) return false

  // Sound safeguard: avoid obvious different-sound containment, e.g. "it" in "with".
  if (large.length >= small.length + 2 && large.includes(small)) return false

  return true
}

const STRONG_START_LETTERS = new Set([
  'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm',
  'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z',
])

function initialSoundPrefixLength(left, right) {
  const a = normalizeWord(left)
  const b = normalizeWord(right)
  if (!a || !b) return 0

  const firstA = a[0]
  const firstB = b[0]
  const vowels = new Set(['a', 'e', 'i', 'o', 'u'])

  // If either starts with a vowel, require first two letters to match.
  if (vowels.has(firstA) || vowels.has(firstB)) {
    if (a.length < 2 || b.length < 2) return 0
    return a.slice(0, 2) === b.slice(0, 2) ? 2 : 0
  }

  // Strong consonant-start match by first letter.
  if (STRONG_START_LETTERS.has(firstA) && STRONG_START_LETTERS.has(firstB)) {
    return firstA === firstB ? 1 : 0
  }

  return 0
}

function extractDescriptivePhrases(sentence) {
  const patterns = [
    /\b\w+(?:\s+\w+){0,3}\s+(?:from|of|into|through|within|under|over)\s+\w+(?:\s+\w+){0,2}\b/gi,
    /\b\w+(?:ed|ing)\s+\w+(?:\s+\w+){0,2}\b/gi,
  ]
  const found = []
  patterns.forEach((pattern) => {
    const matches = sentence.match(pattern) || []
    matches.forEach((match) => found.push(match))
  })
  return found
}

function hexToRgba(hex, alpha = 0.5) {
  if (!hex || typeof hex !== 'string') return `rgba(58, 126, 104, ${alpha})`
  const normalized = hex.replace('#', '').trim()
  if (normalized.length !== 6) return `rgba(58, 126, 104, ${alpha})`
  const r = Number.parseInt(normalized.slice(0, 2), 16)
  const g = Number.parseInt(normalized.slice(2, 4), 16)
  const b = Number.parseInt(normalized.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

function buildSentenceSegments(originalText, sentenceProfiles) {
  if (!originalText) return []
  if (!sentenceProfiles?.length) return [{ type: 'text', text: originalText }]

  const sortedProfiles = sentenceProfiles.slice().sort((a, b) => a.index - b.index)
  const segments = []
  let cursor = 0

  sortedProfiles.forEach((profile) => {
    const sentence = profile.sentence
    if (!sentence) return

    const foundAt = originalText.indexOf(sentence, cursor)
    if (foundAt === -1) return

    if (foundAt > cursor) {
      segments.push({ type: 'text', text: originalText.slice(cursor, foundAt) })
    }

    segments.push({ type: 'sentence', text: sentence, index: profile.index })
    cursor = foundAt + sentence.length
  })

  if (cursor < originalText.length) {
    segments.push({ type: 'text', text: originalText.slice(cursor) })
  }

  return segments
}

function TextConnectionMap({ analysis, originalText, palette }) {
  const canvasRef = useRef(null)
  const textLayerRef = useRef(null)
  const sentenceRefs = useRef({})
  const wordRefs = useRef({})
  const [points, setPoints] = useState({})
  const [isExporting, setIsExporting] = useState(false)

  function anchorFactorForIndex(index) {
    const seed = (Number(index) * 73 + 19) % 100
    return 0.2 + (seed / 100) * 0.6
  }

  const sentences = useMemo(() => {
    if (analysis?.sentence_profiles?.length) {
      return analysis.sentence_profiles
        .slice()
        .sort((a, b) => a.index - b.index)
        .map((item) => item.sentence)
    }
    return splitSentences(originalText)
  }, [analysis, originalText])

  const sentenceSegments = useMemo(
    () => buildSentenceSegments(originalText, analysis?.sentence_profiles ?? []),
    [originalText, analysis],
  )

  const renderableSegments = useMemo(() => {
    const tokenCursor = {}
    const segments = []

    sentenceSegments.forEach((segment, segmentIndex) => {
      if (segment.type !== 'sentence') {
        segments.push({ type: 'text', text: segment.text, key: `text-${segmentIndex}` })
        return
      }

      const sentenceIndex = segment.index
      tokenCursor[sentenceIndex] = tokenCursor[sentenceIndex] || 0
      const chunks = segment.text.split(/(\b[\w']+\b)/g)
      const parts = chunks
        .filter((chunk) => chunk !== '')
        .map((chunk, chunkIndex) => {
          if (/^[\w']+$/.test(chunk)) {
            const tokenIndex = tokenCursor[sentenceIndex]
              const norm = normalizeWord(chunk)
              if (!norm) {
                return {
                  type: 'punct',
                  text: chunk,
                  key: `skip-${sentenceIndex}-${tokenIndex}-${chunkIndex}`,
                }
              }
            tokenCursor[sentenceIndex] += 1
            return {
              type: 'word',
              text: chunk,
                norm,
              sentenceIndex,
              tokenIndex,
              tokenId: `${sentenceIndex}:${tokenIndex}`,
              key: `word-${sentenceIndex}-${tokenIndex}-${chunkIndex}`,
            }
          }

          return {
            type: 'punct',
            text: chunk,
            key: `punct-${segmentIndex}-${chunkIndex}`,
          }
        })

      segments.push({
        type: 'sentence',
        sentenceIndex,
        key: `sentence-${sentenceIndex}-${segmentIndex}`,
        parts,
      })
    })

    return segments
  }, [sentenceSegments])

  const allTokens = useMemo(() => {
    const tokens = []
    renderableSegments.forEach((segment) => {
      if (segment.type !== 'sentence') return
      segment.parts.forEach((part) => {
        if (part.type === 'word' && part.norm) {
          tokens.push(part)
        }
      })
    })
    return tokens
  }, [renderableSegments])

  const tokenIdsBySentenceNorm = useMemo(() => {
    const table = {}
    renderableSegments.forEach((segment) => {
      if (segment.type !== 'sentence') return
      segment.parts.forEach((part) => {
        if (part.type !== 'word' || !part.norm) return
        const sentenceBucket = (table[part.sentenceIndex] ||= {})
        const list = (sentenceBucket[part.norm] ||= [])
        list.push(part.tokenId)
      })
    })
    return table
  }, [renderableSegments])

  const edges = useMemo(() => {
    const relationRank = {
      idea_sentence_arrow: 7,
      punchline_repeat: 6,
      metaphor_bridge: 5,
      same_word_variation: 5,
      sound_similarity: 4,
      small_word_overlap: 4,
      similar_idea: 3,
      important_point_bridge: 2,
    }

    const sentenceProfiles = analysis?.sentence_profiles || []
    const sentenceTexts = sentenceProfiles.length
      ? sentenceProfiles.slice().sort((a, b) => a.index - b.index).map((s) => s.sentence)
      : splitSentences(originalText)

    const tokenizedSentences = sentenceTexts.map((sentence, sentenceIndex) =>
      (sentence.match(/\b[\w']+\b/g) || []).map((surface, tokenIndex) => ({
        sentenceIndex,
        tokenIndex,
        surface,
        norm: normalizeWord(surface),
        stem: stemWord(surface),
        sound: soundex(surface),
      })),
    )

    const flatTokens = tokenizedSentences.flat().filter(
      (item) => item.norm && !VIS_STOPWORDS.has(item.norm),
    )

    const tokenCounts = {}
    flatTokens.forEach((token) => {
      tokenCounts[token.norm] = (tokenCounts[token.norm] || 0) + 1
    })

    const punchlineScores = {}
    Object.entries(tokenCounts).forEach(([word, count]) => {
      let score = 0
      if (count >= 2) score += 1.8
      if (EMO_HINT_WORDS.has(word)) score += 2.2
      if ((analysis?.keywords || []).includes(word)) score += 1.1
      if (word.length >= 6) score += 0.4
      punchlineScores[word] = score
    })

    const punchlineSet = new Set(
      Object.entries(punchlineScores)
        .filter(([, score]) => score >= 1.8)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 40)
        .map(([word]) => word),
    )

    const occurrencesByNorm = {}
    flatTokens.forEach((token) => {
      const list = (occurrencesByNorm[token.norm] ||= [])
      list.push(token)
    })

    const semanticLinks = analysis?.semantic_links || []
    const semanticBySentencePair = new Map()
    semanticLinks.forEach((link) => {
      semanticBySentencePair.set(
        `${link.sentence_a_index}-${link.sentence_b_index}`,
        link,
      )
    })

    const sentenceNormSets = tokenizedSentences.map((tokens) =>
      new Set(tokens.map((token) => token.norm).filter(Boolean)),
    )

    const bestByKey = new Map()

    function addLink(link) {
      const pairKey = [
        Math.min(link.a, link.b),
        Math.max(link.a, link.b),
        link.leftWord,
        link.rightWord,
        link.relation,
      ].join('-')
      const existing = bestByKey.get(pairKey)
      if (!existing || link.strength > existing.strength) {
        bestByKey.set(pairKey, { ...link, key: pairKey })
      }
    }

    // 1) Repeated punchline words.
    Object.entries(occurrencesByNorm).forEach(([word, occurrences]) => {
      if (!punchlineSet.has(word) || occurrences.length < 2) return
      for (let i = 0; i < occurrences.length - 1; i += 1) {
        const left = occurrences[i]
        const right = occurrences[i + 1]
        if (left.sentenceIndex === right.sentenceIndex) continue
        addLink({
          a: left.sentenceIndex,
          b: right.sentenceIndex,
          leftWord: left.surface,
          rightWord: right.surface,
          relation: 'punchline_repeat',
          strength: 0.95 + Math.min(0.4, (punchlineScores[word] || 0) * 0.08),
        })
      }
    })

    // 2) Metaphor/simile-by-description bridges (without looking for "like/as").
    const descriptiveBySentence = sentenceTexts.map((sentence) =>
      extractDescriptivePhrases(sentence).map((phrase) => ({
        phrase,
        words: (phrase.match(/\b[\w']+\b/g) || []).map(normalizeWord).filter(Boolean),
      })),
    )

    for (let i = 0; i < descriptiveBySentence.length; i += 1) {
      for (let j = i + 1; j < descriptiveBySentence.length; j += 1) {
        const leftPhrases = descriptiveBySentence[i]
        const rightPhrases = descriptiveBySentence[j]
        if (!leftPhrases.length || !rightPhrases.length) continue

        const semantic =
          semanticBySentencePair.get(`${i}-${j}`) ||
          semanticBySentencePair.get(`${j}-${i}`)
        const hasMeaningSupport = (semantic?.meaning_strength || 0) >= 0.22

        leftPhrases.forEach((leftPhrase) => {
          rightPhrases.forEach((rightPhrase) => {
            const overlap = leftPhrase.words.filter((word) => rightPhrase.words.includes(word))
            if (!hasMeaningSupport && overlap.length === 0) return

            const leftWord = leftPhrase.words.find((word) => !VIS_STOPWORDS.has(word)) || leftPhrase.words[0]
            const rightWord = rightPhrase.words.find((word) => !VIS_STOPWORDS.has(word)) || rightPhrase.words[0]
            if (!leftWord || !rightWord || leftWord === rightWord) return

            addLink({
              a: i,
              b: j,
              leftWord,
              rightWord,
              relation: 'metaphor_bridge',
              strength: 0.72 + Math.min(0.28, (semantic?.meaning_strength || 0) * 0.5),
            })
          })
        })
      }
    }

    // 3) Highlighting via new rule types:
    // Rule 3 (light red): phonetic signature (same Soundex)
    // Rule 4 (blue): rhythm (same syllables + similar letters)
    // Rule 5 (dark red underline): rhyming (similar phonetic endings)

    // 4) Morphological variation.
    const occurrencesByStem = {}
    flatTokens.forEach((token) => {
      if (!token.stem || token.stem.length < 3) return
      const list = (occurrencesByStem[token.stem] ||= [])
      list.push(token)
    })

    Object.values(occurrencesByStem).forEach((items) => {
      if (items.length < 2) return
      for (let i = 0; i < items.length - 1; i += 1) {
        for (let j = i + 1; j < items.length; j += 1) {
          const left = items[i]
          const right = items[j]
          if (left.sentenceIndex === right.sentenceIndex) continue
          if (left.norm === right.norm) continue
          addLink({
            a: left.sentenceIndex,
            b: right.sentenceIndex,
            leftWord: left.surface,
            rightWord: right.surface,
            relation: 'same_word_variation',
            strength: 0.78,
          })
        }
      }
    })

    // 5) Small word (2-3 chars) connects to larger word when >=50% letters overlap,
    // as long as the pair doesn't look like a different-sound containment.
    const allTokensForSmallRule = tokenizedSentences.flat().filter((item) => item.norm)
    for (let i = 0; i < allTokensForSmallRule.length; i += 1) {
      for (let j = i + 1; j < allTokensForSmallRule.length; j += 1) {
        const left = allTokensForSmallRule[i]
        const right = allTokensForSmallRule[j]
        // Only connect when sentences are directly adjacent.
        if (Math.abs(left.sentenceIndex - right.sentenceIndex) !== 1) continue
        if (!smallWordHalfOverlapConnects(left.norm, right.norm)) continue

        addLink({
          a: left.sentenceIndex,
          b: right.sentenceIndex,
          leftWord: left.surface,
          rightWord: right.surface,
          relation: 'small_word_overlap',
          strength: 0.66,
        })
      }
    }

    // 6) Similar idea clusters.
    for (let i = 0; i < flatTokens.length; i += 1) {
      for (let j = i + 1; j < flatTokens.length; j += 1) {
        const left = flatTokens[i]
        const right = flatTokens[j]
        if (left.sentenceIndex === right.sentenceIndex) continue
        if (!isIdeaRelated(left.norm, right.norm)) continue
        addLink({
          a: left.sentenceIndex,
          b: right.sentenceIndex,
          leftWord: left.surface,
          rightWord: right.surface,
          relation: 'similar_idea',
          strength: 0.69,
        })
      }
    }

    // 7) Important-point bridges via semantic link reasons and important words.
    semanticLinks.forEach((link) => {
      const reasons = link.reasons || []
      reasons.forEach((reason) => {
        const match = reason.match(/([a-z']+)~([a-z']+)/gi)
        if (!match) return
        match.forEach((pairText) => {
          const [leftWordRaw, rightWordRaw] = pairText.toLowerCase().split('~')
          if (!leftWordRaw || !rightWordRaw) return
          if (
            !punchlineSet.has(leftWordRaw) &&
            !punchlineSet.has(rightWordRaw) &&
            !EMO_HINT_WORDS.has(leftWordRaw) &&
            !EMO_HINT_WORDS.has(rightWordRaw)
          ) {
            return
          }

          addLink({
            a: link.sentence_a_index,
            b: link.sentence_b_index,
            leftWord: leftWordRaw,
            rightWord: rightWordRaw,
            relation: 'important_point_bridge',
            strength: 0.64 + Math.min(0.22, (link.meaning_strength || 0) * 0.35),
          })
        })
      })
    })

    // 8) Sentence-level abstract idea arrows between the most relevant words.
    semanticLinks.forEach((link) => {
      const a = Number(link.sentence_a_index)
      const b = Number(link.sentence_b_index)
      if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return
      const earlier = Math.min(a, b)
      const later = Math.max(a, b)
      const meaningStrength = link.meaning_strength || 0
      if (meaningStrength < 0.2) return

      const reasons = link.reasons || []
      const parsedPairs = []
      reasons.forEach((reason) => {
        const match = reason.match(/([a-z']+)~([a-z']+)/gi)
        if (!match) return
        match.forEach((pairText) => {
          const [leftWordRaw, rightWordRaw] = pairText.toLowerCase().split('~')
          if (leftWordRaw && rightWordRaw) {
            parsedPairs.push([leftWordRaw, rightWordRaw])
          }
        })
      })

      const orientedPairs =
        a <= b ? parsedPairs : parsedPairs.map(([leftWordRaw, rightWordRaw]) => [rightWordRaw, leftWordRaw])

      const chosenPair = orientedPairs.find(
        ([leftWordRaw, rightWordRaw]) =>
          sentenceNormSets[earlier]?.has(leftWordRaw) && sentenceNormSets[later]?.has(rightWordRaw),
      )

      const fallbackLeft = tokenizedSentences[earlier]?.find((token) => token.norm && !VIS_STOPWORDS.has(token.norm))
      const fallbackRight = tokenizedSentences[later]?.find((token) => token.norm && !VIS_STOPWORDS.has(token.norm))

      const leftWord = chosenPair?.[0] || fallbackLeft?.norm
      const rightWord = chosenPair?.[1] || fallbackRight?.norm
      if (!leftWord || !rightWord) return

      addLink({
        a: earlier,
        b: later,
        leftWord,
        rightWord,
        relation: 'idea_sentence_arrow',
        strength: 0.72 + Math.min(0.24, meaningStrength * 0.35),
      })
    })

    return [...bestByKey.values()]
      .sort(
        (a, b) =>
          (relationRank[b.relation] || 0) - (relationRank[a.relation] || 0) ||
          b.strength - a.strength,
      )
      .slice(0, 180)
  }, [analysis, originalText])

  function normalizedWord(word) {
    return normalizeWord(word)
  }

  const smallWordUnderlineLetters = useMemo(() => {
    const byTokenId = {}

    edges
      .filter((edge) => edge.relation === 'small_word_overlap')
      .forEach((edge) => {
        const leftNorm = stripSilentEndingE(edge.leftWord)
        const rightNorm = stripSilentEndingE(edge.rightWord)
        if (!leftNorm || !rightNorm) return

        const small = leftNorm.length <= rightNorm.length ? leftNorm : rightNorm
        const large = leftNorm.length <= rightNorm.length ? rightNorm : leftNorm
        const shared = [...new Set(small.split(''))].filter((ch) => large.includes(ch))
        if (!shared.length) return

        const leftTokenIds = tokenIdsBySentenceNorm[edge.a]?.[leftNorm] ?? []
        const rightTokenIds = tokenIdsBySentenceNorm[edge.b]?.[rightNorm] ?? []

        ;[...leftTokenIds, ...rightTokenIds].forEach((tokenId) => {
          const set = byTokenId[tokenId] || new Set()
          shared.forEach((ch) => set.add(ch))
          byTokenId[tokenId] = set
        })
      })

    return byTokenId
  }, [edges, tokenIdsBySentenceNorm])

  const initialSoundUnderlinePrefixByTokenId = useMemo(() => {
    const map = {}

    for (let i = 0; i < allTokens.length; i += 1) {
      for (let j = i + 1; j < allTokens.length; j += 1) {
        const left = allTokens[i]
        const right = allTokens[j]
        if (Math.abs(left.sentenceIndex - right.sentenceIndex) > 2) continue
        const prefixLength = initialSoundPrefixLength(left.norm, right.norm)
        if (!prefixLength) continue

        map[left.tokenId] = Math.max(map[left.tokenId] || 0, prefixLength)
        map[right.tokenId] = Math.max(map[right.tokenId] || 0, prefixLength)
      }
    }

    return map
  }, [allTokens])

  function registerWordRef(tokenId, element) {
    if (!element) return
    wordRefs.current[tokenId] = element
  }

  function pointFromRect(rect, containerRect) {
    return {
      x: rect.left - containerRect.left + rect.width / 2,
      y: rect.top - containerRect.top + rect.height / 2,
    }
  }

  function measureSentencePoints() {
    if (!canvasRef.current) return
    const containerRect = canvasRef.current.getBoundingClientRect()
    const nextPoints = { sentenceCenters: {}, sentenceAnchors: {}, tokenCenters: {}, links: {} }

    Object.entries(sentenceRefs.current).forEach(([index, element]) => {
      if (!element) return
      const rect = element.getBoundingClientRect()
      const factor = anchorFactorForIndex(index)
      nextPoints.sentenceCenters[index] = {
        x: rect.left - containerRect.left + rect.width * factor,
        y: rect.top - containerRect.top + rect.height / 2,
      }
      nextPoints.sentenceAnchors[index] = {
        start: {
          x: rect.left - containerRect.left + 2,
          y: rect.top - containerRect.top + rect.height / 2,
        },
        end: {
          x: rect.right - containerRect.left - 2,
          y: rect.top - containerRect.top + rect.height / 2,
        },
      }
    })

    Object.entries(wordRefs.current).forEach(([tokenId, element]) => {
      if (!element) return
      nextPoints.tokenCenters[tokenId] = pointFromRect(
        element.getBoundingClientRect(),
        containerRect,
      )
    })

    function closestPair(leftCandidates, rightCandidates) {
      let best = null
      leftCandidates.forEach((left) => {
        rightCandidates.forEach((right) => {
          const dx = right.x - left.x
          const dy = right.y - left.y
          const d2 = dx * dx + dy * dy
          if (!best || d2 < best.d2) {
            best = { left, right, d2 }
          }
        })
      })
      return best
    }

    edges.forEach((edge) => {
      if (edge.relation === 'idea_sentence_arrow') {
        const leftWord = normalizedWord(edge.leftWord)
        const rightWord = normalizedWord(edge.rightWord)
        const leftTokenIds = tokenIdsBySentenceNorm[edge.a]?.[leftWord] ?? []
        const rightTokenIds = tokenIdsBySentenceNorm[edge.b]?.[rightWord] ?? []

        const leftCandidates = leftTokenIds
          .map((tokenId) => nextPoints.tokenCenters[tokenId])
          .filter(Boolean)
        const rightCandidates = rightTokenIds
          .map((tokenId) => nextPoints.tokenCenters[tokenId])
          .filter(Boolean)

        if (leftCandidates.length && rightCandidates.length) {
          const best = closestPair(leftCandidates, rightCandidates)
          if (best) {
            nextPoints.links[edge.key] = { start: best.left, end: best.right }
            return
          }
        }

        // Fallback only when relevant tokens cannot be resolved.
        const startAnchor = nextPoints.sentenceAnchors[edge.a]?.end
        const endAnchor = nextPoints.sentenceAnchors[edge.b]?.start
        if (startAnchor && endAnchor) {
          nextPoints.links[edge.key] = { start: startAnchor, end: endAnchor }
        }
        return
      }

      const leftWord = normalizedWord(edge.leftWord)
      const rightWord = normalizedWord(edge.rightWord)
      const leftTokenIds = tokenIdsBySentenceNorm[edge.a]?.[leftWord] ?? []
      const rightTokenIds = tokenIdsBySentenceNorm[edge.b]?.[rightWord] ?? []

      const leftCandidates = leftTokenIds
        .map((tokenId) => nextPoints.tokenCenters[tokenId])
        .filter(Boolean)
      const rightCandidates = rightTokenIds
        .map((tokenId) => nextPoints.tokenCenters[tokenId])
        .filter(Boolean)

      if (!leftCandidates.length || !rightCandidates.length) {
        return
      }

      const best = closestPair(leftCandidates, rightCandidates)
      if (best) {
        nextPoints.links[edge.key] = { start: best.left, end: best.right }
      }
    })

    setPoints(nextPoints)
  }

  function fitTextToCanvas() {
    if (!textLayerRef.current) return
    const content = textLayerRef.current

    let size = Math.max(11, Math.min(23, content.clientWidth / 34))
    content.style.fontSize = `${size}px`
    content.style.lineHeight = '1.48'

    let guard = 0
    while (
      (content.scrollHeight > content.clientHeight ||
        content.scrollWidth > content.clientWidth) &&
      size > 9 &&
      guard < 80
    ) {
      size -= 0.4
      content.style.fontSize = `${size}px`
      guard += 1
    }
  }

  function recalcLayout() {
    fitTextToCanvas()
    requestAnimationFrame(() => {
      measureSentencePoints()
    })
  }

  useLayoutEffect(() => {
    sentenceRefs.current = {}
    wordRefs.current = {}
    recalcLayout()

    window.addEventListener('resize', recalcLayout)
    return () => window.removeEventListener('resize', recalcLayout)
  }, [renderableSegments, edges])

  useEffect(() => {
    if (!canvasRef.current) return

    const observer = new ResizeObserver(recalcLayout)
    observer.observe(canvasRef.current)
    return () => observer.disconnect()
  }, [renderableSegments])

  useEffect(() => {
    if (typeof document === 'undefined' || !document.fonts?.ready) return
    let cancelled = false
    document.fonts.ready.then(() => {
      if (!cancelled) {
        recalcLayout()
      }
    })
    return () => {
      cancelled = true
    }
  }, [renderableSegments, edges])

  async function downloadPng() {
    if (!canvasRef.current) return

    try {
      setIsExporting(true)
      const canvas = await html2canvas(canvasRef.current, {
        backgroundColor: '#fff8f0',
        scale: 2,
      })
      const imageUrl = canvas.toDataURL('image/png')
      const anchor = document.createElement('a')
      anchor.href = imageUrl
      anchor.download = 'text-connection-map.png'
      anchor.click()
    } finally {
      setIsExporting(false)
    }
  }

  const maxStrength = Math.max(...edges.map((edge) => edge.strength || 0), 1)
  const linkPoints = Object.values(points.links || {})
  const yValues = linkPoints.flatMap((link) => [link.start.y, link.end.y])
  const textTop = yValues.length ? Math.min(...yValues) : 0
  const textBottom = yValues.length ? Math.max(...yValues) : 0

  if (!sentences.length) {
    return <p className="hint">Run analysis to see text connections.</p>
  }

  return (
    <div className="literal-map-wrap">
      <button
        type="button"
        onClick={downloadPng}
        className="export-btn"
        disabled={isExporting}
      >
        {isExporting ? 'Exporting PNG...' : 'Download PNG'}
      </button>

      <div className="literal-map-canvas" ref={canvasRef}>
        <svg className="connection-lines" aria-hidden="true">
          <defs>
            <marker
              id="idea-arrowhead"
              markerWidth="4"
              markerHeight="3"
              refX="3.6"
              refY="1.5"
              orient="auto"
            >
              <path d="M0,0 L4,1.5 L0,3 Z" fill="rgba(40, 40, 40, 0.72)" />
            </marker>
          </defs>
          {edges.filter((edge) => edge.relation !== 'small_word_overlap').map((edge, index) => {
            const linked = points.links?.[edge.key]
            const start = linked?.start
            const end = linked?.end
            if (!start || !end) return null

            const dx = end.x - start.x
            const dy = end.y - start.y
            const distance = Math.max(Math.hypot(dx, dy), 1)
            const nx = -dy / distance
            const ny = dx / distance

            const bendDirection = (edge.a + edge.b + index) % 2 === 0 ? 1 : -1

            // Closer links bend more; farther links stay closer to straight.
            const curvatureStrength = Math.max(0, Math.min(1, 16000 / (distance * distance + 16000)))

            // Make the near full-height (top-to-bottom) connection the straightest.
            const totalVerticalSpan = Math.max(1, textBottom - textTop)
            const edgeVerticalSpan = Math.abs(end.y - start.y)
            const topToBottomRatio = Math.max(0, Math.min(1, edgeVerticalSpan / totalVerticalSpan))
            const isTopToBottomLine = topToBottomRatio >= 0.98

            const offset = isTopToBottomLine
              ? 1
              : 4 + curvatureStrength * 22

            const cx = (start.x + end.x) / 2 + nx * offset * bendDirection
            const rawCy = (start.y + end.y) / 2 + ny * offset * bendDirection
            const cy = Math.min(
              Math.max(rawCy, textTop + 6),
              textBottom - 6,
            )

            const baseWidth = 1 + (edge.strength / maxStrength) * 3
            const lineColor = hexToRgba(
              palette?.[index % (palette?.length || 1)] || '#3a7e68',
              0.5,
            )
            const strokeColor = edge.relation === 'idea_sentence_arrow' ? 'rgba(40, 40, 40, 0.72)' : lineColor
            const markerEnd = edge.relation === 'idea_sentence_arrow' ? 'url(#idea-arrowhead)' : undefined
            const width = edge.relation === 'idea_sentence_arrow' ? Math.max(0.6, baseWidth * 0.35) : baseWidth

            return (
              <path
                key={edge.key || `edge-${index}-${edge.a}-${edge.b}`}
                d={`M ${start.x} ${start.y} Q ${cx} ${cy} ${end.x} ${end.y}`}
                stroke={strokeColor}
                strokeWidth={width}
                markerEnd={markerEnd}
                fill="none"
              />
            )
          })}
        </svg>

        <div
          ref={textLayerRef}
          className="capture-text-layer"
        >
          {renderableSegments.map((segment) => {
            if (segment.type === 'sentence') {
              return (
                <span
                  key={segment.key}
                  ref={(element) => {
                    sentenceRefs.current[segment.sentenceIndex] = element
                  }}
                  className="sentence-anchor"
                >
                  {segment.parts.map((part) => {
                    if (part.type === 'word') {
                      const underlineSet = smallWordUnderlineLetters[part.tokenId]
                      const hasUnderline = underlineSet && underlineSet.size > 0
                      const prefixUnderlineLength = initialSoundUnderlinePrefixByTokenId[part.tokenId] || 0
                      let leadingLetterCount = 0
                      const renderedWord = part.text.split('').map((char, idx) => {
                        const isLetter = /[a-z]/i.test(char)
                        const underlineBySmallRule = isLetter && hasUnderline && underlineSet.has(char.toLowerCase())

                        let underlineByInitialRule = false
                        if (isLetter && leadingLetterCount < prefixUnderlineLength) {
                          underlineByInitialRule = true
                          leadingLetterCount += 1
                        }

                        const style = underlineByInitialRule
                          ? {
                              textDecorationLine: 'underline',
                              textDecorationColor: 'rgba(0, 102, 204, 0.3)',
                              textDecorationThickness: '2px',
                            }
                          : underlineBySmallRule
                            ? {
                                textDecorationLine: 'underline',
                                textDecorationColor: 'rgba(139, 0, 0, 0.3)',
                                textDecorationThickness: '2px',
                              }
                            : undefined

                        return (
                          <span key={`${part.key}-${idx}`} style={style}>
                            {char}
                          </span>
                        )
                      })
                      return (
                        <span
                          key={part.key}
                          ref={(element) => {
                            registerWordRef(part.tokenId, element)
                          }}
                          className="word-anchor"
                        >
                          {renderedWord}
                        </span>
                      )
                    }
                    return <span key={part.key}>{part.text}</span>
                  })}
                </span>
              )
            }

            return <span key={segment.key}>{segment.text}</span>
          })}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [text, setText] = useState('')
  const [photo, setPhoto] = useState(null)
  const [paletteMode, setPaletteMode] = useState('clothing')
  const [loading, setLoading] = useState(false)
  const [generatingImage, setGeneratingImage] = useState(false)
  const [error, setError] = useState('')
  const [imageError, setImageError] = useState('')
  const [result, setResult] = useState(null)
  const [generatedImage, setGeneratedImage] = useState(null)
  const [analyzedSignature, setAnalyzedSignature] = useState('')

  const apiBase =
    import.meta.env.VITE_API_BASE?.trim() ||
    (typeof window !== 'undefined' ? window.location.origin : 'http://127.0.0.1:8000')

  const palette = useMemo(() => result?.image_analysis?.palette ?? [], [result])
  const topDisplayColors = useMemo(() => {
    const picked = []
    const sources = [
      generatedImage?.portrait_top_colors ?? [],
      result?.image_analysis?.clothing_palette ?? [],
      result?.image_analysis?.full_image_palette ?? [],
      palette,
    ]

    for (const source of sources) {
      for (const value of source) {
        const hex = String(value || '').trim().toUpperCase()
        if (!/^#[0-9A-F]{6}$/.test(hex)) continue
        if (picked.includes(hex)) continue
        picked.push(hex)
        if (picked.length >= 5) return picked
      }
    }

    const fallback = '#3A7E68'
    while (picked.length < 5) {
      picked.push(picked[picked.length - 1] || fallback)
    }

    return picked
  }, [generatedImage, result, palette])

  function conceptualReasonTwoWords(link) {
    const reasons = link?.reasons || []
    const bannedWords = new Set()

    reasons.forEach((reason) => {
      const pairMatches = reason.match(/([a-z']+)~([a-z']+)/gi) || []
      pairMatches.forEach((pairText) => {
        const [leftRaw, rightRaw] = pairText.toLowerCase().split('~')
        const left = normalizeWord(leftRaw)
        const right = normalizeWord(rightRaw)
        if (left) bannedWords.add(left)
        if (right) bannedWords.add(right)
      })
    })

    const sentenceATokens = (link?.sentence_a?.match(/\b[\w']+\b/g) || [])
      .map((word) => normalizeWord(word))
      .filter((word) => word && !VIS_STOPWORDS.has(word) && !bannedWords.has(word))
    const sentenceBTokens = (link?.sentence_b?.match(/\b[\w']+\b/g) || [])
      .map((word) => normalizeWord(word))
      .filter((word) => word && !VIS_STOPWORDS.has(word) && !bannedWords.has(word))

    const sentenceBSet = new Set(sentenceBTokens)
    const sentenceASet = new Set(sentenceATokens)

    function bestToken(tokens, otherSet) {
      const scored = tokens
        .filter((word) => word.length >= 3)
        .map((word, idx) => ({
          word,
          // Prefer concrete words, repeated motifs, and earlier salient mentions.
          score:
            (word.length >= 6 ? 2 : 0) +
            (otherSet.has(word) ? 2 : 0) +
            Math.max(0, 1 - idx * 0.08),
        }))
        .sort((a, b) => b.score - a.score)
      return scored[0]?.word || ''
    }

    const left = bestToken(sentenceATokens, sentenceBSet)
    let right = bestToken(sentenceBTokens, sentenceASet)

    if (!left && !right) {
      return 'idea linkage'
    }

    if (!left) {
      const altLeft = sentenceATokens.find((word) => word !== right) || right
      return `${altLeft} ${right}`
    }

    if (!right) {
      const altRight = sentenceBTokens.find((word) => word !== left) || left
      return `${left} ${altRight}`
    }

    if (left === right) {
      const altRight = sentenceBTokens.find((word) => word !== left) || right
      return `${left} ${altRight}`
    }

    return `${left} ${right}`
  }

  const contextualIdeaLinks = useMemo(() => {
    const links = result?.text_analysis?.semantic_links ?? []
    return links
      .filter((link) => (link.meaning_strength || 0) >= 0.16)
      .sort((a, b) => (b.meaning_strength || 0) - (a.meaning_strength || 0))
      .slice(0, 12)
      .map((link) => ({
        ...link,
        concept_reason: conceptualReasonTwoWords(link),
      }))
  }, [result])

  function currentInputSignature() {
    const photoSig = photo
      ? `${photo.name || 'photo'}:${photo.size || 0}:${photo.lastModified || 0}`
      : 'no-photo'
    return `${text.trim()}||${paletteMode}||${photoSig}`
  }

  async function handleGenerateEmotionImage() {
    if (!result?.text_analysis) return
    if (!photo) {
      setImageError('Please upload a photo first so the exact subject can be preserved.')
      return
    }

    if (currentInputSignature() !== analyzedSignature) {
      setImageError('Inputs changed after analysis. Please click "Analyze Text + Photo" again before generating.')
      return
    }

    setImageError('')

    try {
      setGeneratingImage(true)
      const formData = new FormData()
      formData.append('photo', photo)
      formData.append('text_analysis', JSON.stringify(result.text_analysis || {}))
      formData.append('palette', JSON.stringify(result?.image_analysis?.palette ?? []))
      formData.append(
        'clothing_palette',
        JSON.stringify(result?.image_analysis?.clothing_palette ?? []),
      )
      formData.append(
        'full_image_palette',
        JSON.stringify(result?.image_analysis?.full_image_palette ?? []),
      )
      formData.append(
        'clothing_style_profile',
        JSON.stringify(result?.image_analysis?.clothing_style_profile ?? {}),
      )
      formData.append('palette_mode', paletteMode)
      formData.append('generation_backend', 'sd3_image_to_image')

      const response = await axios.post(`${apiBase}/generate-emotion-image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setGeneratedImage(response.data)
    } catch (requestError) {
      const status = requestError?.response?.status
      const detail = requestError?.response?.data?.detail
      if (status && detail) {
        setImageError(`Error ${status}: ${detail}`)
      } else {
        setImageError('Image generation failed. Check the backend Stability AI key.')
      }
    } finally {
      setGeneratingImage(false)
    }
  }

  const onSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setImageError('')

    if (!text.trim()) {
      setError('Please add some text first.')
      return
    }

    if (!photo) {
      setError('Please upload a photo.')
      return
    }

    const formData = new FormData()
    formData.append('text', text)
    formData.append('photo', photo)

    try {
      setLoading(true)
      setGeneratedImage(null)
      const response = await axios.post(`${apiBase}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(response.data)
      setAnalyzedSignature(currentInputSignature())
    } catch (requestError) {
      const status = requestError?.response?.status
      const detail = requestError?.response?.data?.detail
      if (status && detail) {
        setError(`Error ${status}: ${detail}`)
      } else {
        setError('Request failed. Make sure backend is running on port 8000.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      {/* Header removed as requested */}
      <section className="panel input-panel">
        <form onSubmit={onSubmit}>
          <label htmlFor="text-input">Text Input</label>
          <textarea
            id="text-input"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Paste poem, lyrics, journal entry, or paragraph..."
            rows={8}
          />

          <label htmlFor="photo-input">Photo Upload</label>
          <input
            id="photo-input"
            type="file"
            accept="image/*"
            onChange={(event) => setPhoto(event.target.files?.[0] ?? null)}
          />

          <label htmlFor="palette-mode">Palette Source</label>
          <select
            id="palette-mode"
            value={paletteMode}
            onChange={(event) => setPaletteMode(event.target.value)}
          >
            <option value="clothing">Clothing Palette</option>
            <option value="full">Full Image Palette</option>
          </select>

          <button type="submit" disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze Text + Photo'}
          </button>
        </form>

        {error ? <p className="error">{error}</p> : null}

        {result ? (
          <div className="left-image-panel">
            <h3>Emotion Image</h3>
            {!generatedImage ? (
              <>
                <p className="hint">
                  Review the analysis first, then generate one emotion-based image for this session.
                </p>
                {currentInputSignature() !== analyzedSignature ? (
                  <p className="hint" style={{ color: '#9c5d00' }}>
                    Inputs changed since the last analysis. Re-run analysis to refresh prompt and colors.
                  </p>
                ) : null}
                <button
                  type="button"
                  onClick={handleGenerateEmotionImage}
                  disabled={generatingImage}
                >
                  {generatingImage ? 'Generating Theatre Image...' : 'Generate One Theatre Image'}
                </button>
              </>
            ) : (
              <p className="hint">One image has already been generated for this session.</p>
            )}

            {imageError ? <p className="error">{imageError}</p> : null}

            {generatedImage?.image_data_url ? (
              <figure className="generated-image-wrap">
                <img
                  src={generatedImage.image_data_url}
                  alt="Emotion-based generation"
                  className="generated-image"
                />
                <figcaption>
                  Driven by emotions: {(generatedImage.emotions || []).join(', ') || 'n/a'}
                </figcaption>
              </figure>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="panel output-panel">
        {!result ? (
          <p className="hint">Run analysis to see themes, emotions, and palette.</p>
        ) : (
          <>
            <div className="chips">
              {topDisplayColors.map((hex, index) => (
                <span key={`${hex}-${index}`} className="chip">
                  <i style={{ backgroundColor: hex }} />
                  {hex}
                </span>
              ))}
            </div>

            <h3>Top Themes</h3>
            <ul>
              {result.text_analysis.themes.map((item) => (
                <li key={item.theme}>
                  {item.theme} ({item.confidence})
                </li>
              ))}
            </ul>

            {/* Layered Theme Triangulation section removed as requested */}

            <h3>Dominant Emotions</h3>
            <ul>
              {result.text_analysis.dominant_emotions
                .filter((item) => Number(item.score) > 0.0001)
                .map((item) => (
                <li key={item.emotion}>
                  {item.emotion} ({item.score})
                </li>
              ))}
            </ul>

            {/* Prompt display removed: no longer shown to user */}

            <h3>Text Connection Map</h3>
            <TextConnectionMap
              analysis={result.text_analysis}
              originalText={text}
              palette={palette}
            />

            <h3>Contextual Idea Connections (Backend)</h3>
            {!contextualIdeaLinks.length ? (
              <p className="hint">No strong contextual idea links were detected for this text.</p>
            ) : (
              <ul className="contextual-list">
                {contextualIdeaLinks.map((link, index) => (
                  <li key={`contextual-${link.sentence_a_index}-${link.sentence_b_index}-${index}`}>
                    <strong>
                      S{link.sentence_a_index + 1} ↔ S{link.sentence_b_index + 1}
                    </strong>{' '}
                    ({(link.meaning_strength || 0).toFixed(3)})
                    <br />
                    Concept: {link.concept_reason}
                    <br />
                    {link.reasons?.length ? `Why: ${link.reasons.join(' | ')}` : 'Why: semantic overlap'}
                    <br />
                    <span className="contextual-snippet">{link.sentence_a}</span>
                    <br />
                    <span className="contextual-snippet">{link.sentence_b}</span>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </section>
    </main>
  )
}

export default App
