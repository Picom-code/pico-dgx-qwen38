#!/usr/bin/env python3
"""mkcorpus.py OUTDIR : round 9, builder A. Writes two prompt sets that share NO prompt with any bench set:
  fit.jsonl    the FITTING corpus for the two-stage draft head's rotation (broad: English prose on other topics, code in ~30 languages,
               16 natural languages, tool-call / agent transcripts in three wire formats, structured output, pasted logs / configs /
               data, rewrite tasks on generated files, long-form, a share with thinking on)
  fresh.jsonl  a second set, NEVER used for fitting (other topics, other tasks, five extra languages): the out-of-fit ruler next to
               the 320 held prompts. Categories chat / code / math / lang / agent / struct.
SOURCE: every prompt is written here (topic lists x request forms, by builder A for this round); no public text is fetched; the
answers are whatever the engine itself writes. Deterministic: no random numbers, the same file every run.
Fields: id, cat, prompt, max_tokens (fit only), think (fit only)."""
import json, os, sys

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
FIT, FRESH = [], []
def fit(cat, p, mt=320, think=False): FIT.append(dict(id=len(FIT), cat=cat, prompt=p, max_tokens=mt, think=think))
def fresh(cat, p): FRESH.append(dict(id=len(FRESH), cat=cat, prompt=p))

# ---------------- 1. English prose: topics x forms (fit and fresh use disjoint topic lists) ----------------
FORMS = ["Explain {t} to a curious adult who has never studied it. Use plain language and concrete examples.",
         "Write a five-paragraph essay arguing for a clear position on {t}.",
         "Write a dialogue between two friends who disagree about {t}. Give each a distinct voice.",
         "Write a newspaper feature article about {t}, with a headline, a lede and quotes from invented experts.",
         "Give a practical step-by-step guide for a beginner dealing with {t}. Number the steps.",
         "Write a short story in which {t} plays a central role. Around 400 words.",
         "List the ten most common misconceptions about {t} and correct each one in two sentences.",
         "Write a formal business memo to senior management about {t}, with a recommendation.",
         "Describe the history of {t} from its beginnings to today, with dates where you know them.",
         "Write a product-style FAQ with eight questions and answers about {t}.",
         "Write a lecture transcript, as spoken by a professor, introducing {t} to first-year students.",
         "Write a critical review, with strengths and weaknesses, of the current state of {t}."]
T_FIT = ["tidal power stations", "the migration of monarch butterflies", "sourdough fermentation", "container shipping and port logistics",
         "the Byzantine court", "volcanic soil and wine growing", "urban beekeeping", "the invention of the printing press",
         "how credit scores are calculated", "glacier retreat in Patagonia", "the rules of cricket", "medieval cathedral construction",
         "coral reef restoration", "the placebo effect", "railway signalling", "the economics of streaming music",
         "the Silk Road caravan trade", "antibiotic resistance", "how elevators are scheduled in tall buildings", "the Mongolian nomadic diet",
         "desalination plants", "jazz improvisation", "the psychology of procrastination", "satellite navigation errors",
         "commercial aviation maintenance checks", "the Dutch polder system", "paper recycling", "chess opening theory",
         "octopus intelligence", "the insurance of fine art", "school lunch programmes", "long-distance trucking regulations",
         "typeface design", "the geology of Iceland", "noise pollution in cities", "the Mughal gardens",
         "home composting", "bridge cable inspection", "contract law for freelancers", "the training of guide dogs",
         "the fall of Constantinople", "cold-chain vaccine transport", "amateur radio", "the etiquette of Japanese tea ceremony",
         "wildfire evacuation planning", "the taxation of inheritance", "deep-sea mining", "the restoration of old film reels"]
T_FRESH = ["lighthouse optics and Fresnel lenses", "the domestication of horses", "how bail works in criminal courts", "truffle hunting",
           "wind-turbine blade recycling", "the Hanseatic League", "city bus route planning", "the science of baking at high altitude",
           "pearl farming", "the history of the metric system", "emergency-room triage", "stone-wall building in rural Ireland",
           "the regulation of drones", "glassblowing", "the Antarctic Treaty", "how libraries decide which books to discard",
           "the care of bonsai trees", "airline overbooking", "the Inca road network", "hearing aids and cochlear implants",
           "rural broadband", "the craft of bookbinding", "earthquake early-warning systems", "the trade in used clothing",
           "marathon pacing strategy", "seed banks", "the acoustics of concert halls", "pension fund investing",
           "tide pools", "the design of board games", "food allergies in schools", "canal locks",
           "the Ottoman postal system", "ski-resort snowmaking", "telescope mirror grinding", "the ethics of zoo breeding programmes",
           "water rights in dry regions", "tram networks", "the tuning of pianos", "lost-luggage tracing"]
for i, t in enumerate(T_FIT): fit("prose", FORMS[i % len(FORMS)].format(t=t))
FORMS_FRESH = ["What should an interested outsider know about {t}? Answer in a few clear paragraphs.",
               "Draft a blog post titled 'What nobody tells you about {t}'.",
               "Imagine a radio interview with someone who has worked in {t} for thirty years. Write the transcript.",
               "Give the arguments on both sides of the biggest open debate around {t}, then say which side convinces you more.",
               "Write a letter to a teenage niece who asked you about {t}; be warm, accurate and a little funny.",
               "Compose a museum wall text plus three exhibit captions on {t}.",
               "Tell an anecdote, real-sounding but invented, that illustrates why {t} matters.",
               "Summarise {t} in exactly seven bullet points, then expand the most surprising one into a paragraph."]
for i, t in enumerate(T_FRESH): fresh("chat", FORMS_FRESH[(i * 3 + 1) % len(FORMS_FRESH)].format(t=t))

# ---------------- 2. code: languages x tasks ----------------
L_FIT = ["Python", "Rust", "TypeScript", "Go", "Java", "C", "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Lua", "Elixir", "OCaml",
         "Zig", "Julia", "R", "Perl", "JavaScript (Node.js)", "Dart", "Haskell", "Clojure", "F#", "Erlang", "Fortran 2008", "Ada",
         "x86-64 assembly (NASM, Linux)", "Common Lisp", "Nim", "Objective-C"]
K_FIT = ["an interval tree with insert and overlap query", "a tokenizer for arithmetic expressions and a recursive-descent evaluator",
         "a fixed-capacity ring buffer with iterator support", "union-find with path compression and union by rank, plus a Kruskal demo",
         "a Bloom filter with configurable false-positive rate", "a function that merges k sorted lists using a heap",
         "a validator for 9x9 sudoku boards and a backtracking solver", "a command-line argument parser supporting flags, options and subcommands",
         "LZ77 compression and decompression of byte strings", "an ISO-8601 date and duration parser without library helpers",
         "a finite state machine for a vending machine, with a transition table", "big-integer addition and multiplication on decimal strings",
         "a glob pattern matcher supporting *, ? and character classes", "a least-squares line fit with residual statistics",
         "an event scheduler with a timer wheel", "a skip list with insert, delete and range scan",
         "a Markdown-to-HTML converter for headings, lists, emphasis and links", "the Game of Life on a torus with a text renderer",
         "a diff of two text files using the longest common subsequence", "a simple key-value store with an append-only log and compaction",
         "an expression simplifier for symbolic polynomials", "a consistent-hash ring with virtual nodes",
         "Huffman coding: build the tree, encode and decode", "a tiny HTTP/1.1 request parser handling chunked bodies"]
for i, lang in enumerate(L_FIT):
    for j in range(2):
        k = K_FIT[(i * 7 + j * 11) % len(K_FIT)]
        tail = ["Include comments and a small test or demo.", "Add error handling and explain the complexity at the end.",
                "Write idiomatic code and unit tests.", "Output only the source file."][(i + j) % 4]
        fit("code", f"Implement {k} in {lang}. {tail}")
INFRA = ["Write a Kubernetes manifest (Deployment, Service, HorizontalPodAutoscaler, ConfigMap) for a three-replica API with health probes.",
         "Write a multi-stage Dockerfile for a Rust web service with a distroless final image, and a docker-compose file with Postgres and Redis.",
         "Write Terraform for an AWS VPC with two public and two private subnets, a NAT gateway and an autoscaling group.",
         "Write a GNU Makefile for a C project with src/, include/ and tests/, automatic dependency files, and debug / release / sanitize targets.",
         "Write a GitHub Actions workflow that builds on Linux, macOS and Windows, caches dependencies, runs tests and uploads artifacts on tags.",
         "Write an nginx configuration with TLS, HTTP/2, gzip, rate limiting and a reverse proxy to two upstreams with health checks.",
         "Write a PowerShell script that inventories installed software on remote machines and exports a CSV report.",
         "Write a SystemVerilog module for a parameterised synchronous FIFO with full / empty flags and a self-checking testbench.",
         "Write a LaTeX document with a title page, two sections, a table, an equation environment and a bibliography of three entries.",
         "Write an HTML page with CSS grid for a responsive photo gallery with a dark-mode toggle, without frameworks.",
         "Write PostgreSQL DDL for an e-commerce order system with constraints and triggers, then five analytical queries using window functions.",
         "Write an Ansible playbook that hardens SSH, sets up a firewall, creates users from a variable list and installs a monitoring agent.",
         "Write a CMakeLists.txt for a C++20 library with tests, install rules and an exported package config.",
         "Write an OpenAPI 3.1 specification in YAML for a booking service with five endpoints, schemas and error responses.",
         "Write a Prometheus alerting rules file and a Grafana dashboard JSON for latency, error rate and saturation.",
         "Write a GraphQL schema for a blogging platform and resolvers in pseudo-code, including pagination and authorization.",
         "Write a protobuf file for a chat service with streaming RPCs, and explain field-number compatibility rules.",
         "Write a jq program and an awk program that each summarise a JSON-lines access log by status code and path prefix.",
         "Write a Vim script plugin that toggles comments for several file types, with a help file.",
         "Write a regular expression (PCRE) for each of: IPv4 address, ISO date, hex colour, semantic version; explain each piece."]
for p in INFRA: fit("code", p)
L_FRESH = ["Python", "Rust", "TypeScript", "Go", "Java", "C", "C++", "C#", "Ruby", "Swift", "Kotlin", "Elixir", "Scala", "PHP", "Lua", "Haskell",
           "Crystal", "Groovy", "D", "Racket", "Bash", "SQL (SQLite dialect)", "MATLAB", "COBOL"]
K_FRESH = ["a segment tree for range-minimum queries with point updates", "a parser for INI configuration files with sections and escapes",
           "a rope data structure for string editing", "topological sort with cycle detection and a readable error",
           "a rate-of-change detector over a sliding window of sensor readings", "a base64 encoder and decoder without library helpers",
           "an undo / redo stack for a text editor buffer", "a calendar function that prints any month given year and month",
           "a simple bank-ledger with double-entry checks", "run a breadth-first search on a maze given as text and print the path",
           "a roman-numeral converter in both directions with validation", "a word-frequency counter that reads standard input and prints the top 20",
           "a matrix determinant by LU decomposition with partial pivoting", "a circular doubly linked list with splice",
           "an elevator controller simulation", "a prime sieve that is segmented to stay within a memory limit",
           "a retry helper with exponential backoff and jitter", "a tiny template engine with {{name}} substitution and loops",
           "a binary min-max heap", "a spell-checker suggesting words within edit distance 2",
           "a script that finds duplicate files by size and checksum", "a schema and queries for a flight-booking database with a recursive query for connections",
           "a numerical integrator comparing trapezoid and Simpson rules", "a payroll report from fixed-width records"]
for lang, k in zip(L_FRESH, K_FRESH): fresh("code", f"Using {lang}, write {k}. Comment the tricky parts and finish with a usage example.")

# ---------------- 3. natural languages (the request itself is in the language) ----------------
NL_FIT = {
 "es": ["Explica cómo funciona el sistema inmunitario humano, con ejemplos cotidianos.", "Escribe un cuento corto sobre un pescador que encuentra un reloj antiguo en su red.",
        "Redacta una carta formal al ayuntamiento pidiendo más carriles para bicicletas."],
 "fr": ["Explique le fonctionnement d'une centrale hydroélectrique, étape par étape.", "Rédige une critique d'un restaurant imaginaire de Lyon, avec ses points forts et ses défauts.",
        "Écris un dialogue entre une grand-mère et son petit-fils à propos des téléphones portables."],
 "de": ["Erkläre, wie eine Wärmepumpe funktioniert, und für welche Häuser sie sich lohnt.", "Schreibe einen Zeitungsartikel über die Eröffnung einer neuen Straßenbahnlinie in einer Kleinstadt.",
        "Verfasse eine höfliche Beschwerde an einen Online-Händler wegen einer beschädigten Lieferung."],
 "zh": ["请详细解释高铁是如何保持高速行驶时的稳定和安全的。", "写一篇关于江南水乡清晨的散文，大约四百字。", "请为一家新开的茶馆写一份开业宣传文案，并列出五种招牌茶。"],
 "ja": ["地震が起きる仕組みと、日本の建物がどのように揺れに備えているかを説明してください。", "京都の秋を旅する三日間の計画を、朝・昼・夜に分けて書いてください。",
        "新入社員に向けた、会議での発言のコツについての社内メールを書いてください。"],
 "pt": ["Explique como se forma um furacão e por que eles perdem força ao chegar à terra.", "Escreva uma crônica sobre uma feira de rua numa manhã de sábado em Salvador.",
        "Redija um e-mail profissional recusando educadamente uma proposta de emprego."],
 "ru": ["Объясните, как работает метро: от туннелей и вентиляции до расписания поездов.", "Напишите короткий рассказ о старом библиотекаре, который находит письмо в забытой книге.",
        "Составьте план подготовки к экзамену по истории на две недели."],
 "it": ["Spiega come si produce l'olio d'oliva, dalla raccolta alla bottiglia.", "Scrivi una recensione di un film immaginario ambientato a Napoli negli anni Sessanta.",
        "Scrivi una lettera a un vicino di casa per proporre un orto condiviso nel cortile."],
 "ko": ["김치가 발효되는 과정을 과학적으로 설명해 주세요.", "비 오는 날 서울의 작은 서점을 배경으로 한 짧은 이야기를 써 주세요.", "신입 개발자를 위한 코드 리뷰 예절 안내문을 작성해 주세요."],
 "ar": ["اشرح كيف تعمل محطات تحلية مياه البحر ولماذا هي مهمة في الخليج العربي.", "اكتب قصة قصيرة عن بائع كتب قديم في سوق دمشق.", "اكتب رسالة رسمية إلى مدير المدرسة تقترح فيها إنشاء نادٍ للعلوم."],
 "hi": ["मानसून कैसे बनता है और भारतीय खेती के लिए यह क्यों ज़रूरी है, विस्तार से समझाइए।", "एक छोटे पहाड़ी गाँव की सुबह पर एक निबंध लिखिए।", "अपने मकान मालिक को किराया देर से देने के बारे में एक विनम्र पत्र लिखिए।"],
 "tr": ["Bir deprem sırasında ve sonrasında yapılması gerekenleri adım adım anlatın.", "İstanbul'da bir vapur yolculuğunu anlatan kısa bir öykü yazın.", "Yeni açılan bir kütüphane için tanıtım yazısı yazın."],
 "nl": ["Leg uit hoe de Deltawerken Nederland tegen overstromingen beschermen."], "pl": ["Wyjaśnij, jak powstaje bursztyn i dlaczego jest tak ważny dla Gdańska."],
 "vi": ["Hãy giải thích cách trồng lúa nước ở đồng bằng sông Cửu Long, từ gieo mạ đến thu hoạch."], "id": ["Jelaskan bagaimana gunung berapi terbentuk dan mengapa Indonesia memiliki begitu banyak."],
}
for lg, ps in NL_FIT.items():
    for p in ps: fit("lang-" + lg, p)
fit("lang-mix", "Translate the following English paragraph into Spanish, French, German and Japanese, one after the other, each under its own heading:\n\nThe night train left the station ten minutes late. A conductor walked through the carriage checking tickets, while outside the last lights of the city gave way to dark fields. An old man by the window unwrapped a sandwich and offered half to the student sitting opposite.")
fit("lang-mix", "Write the same short product announcement for a reusable water bottle in English, Portuguese, Korean and Arabic.")
NL_FRESH = [
 "Explica las causas de la sequía en el Mediterráneo y qué pueden hacer las ciudades para ahorrar agua.", "Escribe una reseña de una novela policíaca imaginaria ambientada en Buenos Aires.",
 "Décris le métier de boulanger en France, de la nuit de travail à l'ouverture de la boutique.", "Rédige un courriel à un collègue pour reporter une réunion et proposer trois nouveaux créneaux.",
 "Beschreibe, wie das deutsche Pfandsystem für Flaschen funktioniert.", "Schreibe eine Kurzgeschichte über eine Lokführerin im Schneesturm.",
 "请解释中国传统二十四节气的由来，并举三个例子说明它们如何指导农事。", "给刚搬到上海的外国朋友写一封信，介绍如何使用地铁和共享单车。",
 "日本の四季それぞれの代表的な行事を紹介する文章を書いてください。", "初めて一人暮らしをする学生に向けて、節約のコツを十個挙げてください。",
 "Descreva como funciona o sistema de transporte por barcas entre Rio de Janeiro e Niterói.", "Объясните, почему Байкал считается уникальным озером.",
 "Racconta la storia del caffè espresso e di come si prepara al bar.", "제주도 해녀의 하루를 설명하는 글을 써 주세요.",
 "اشرح كيف يُصنع الخبز العربي التقليدي في الفرن.", "दिल्ली मेट्रो ने शहर के जीवन को कैसे बदला, इस पर एक लेख लिखिए।",
 "Beskriv hur allemansrätten fungerar i Sverige och vilka skyldigheter den medför.", "Opisz tradycje wigilijne w Polsce i ich pochodzenie.",
 "Kerro, miten sauna kuuluu suomalaiseen arkeen ja mitä tapoja siihen liittyy.", "Περιγράψτε πώς λειτουργεί ένα παραδοσιακό ελαιοτριβείο στην Κρήτη.",
 "Popište, jak se vaří české pivo, od sladu po čepování.", "Eleza jinsi kahawa inavyolimwa na kusindikwa katika nyanda za juu za Tanzania.",
 "อธิบายวิธีทำต้มยำกุ้งแบบดั้งเดิม พร้อมเคล็ดลับการเลือกวัตถุดิบ", "הסבר כיצד פועלת השקיה בטפטוף ומדוע היא חוסכת מים."]
for p in NL_FRESH: fresh("lang", p)

# ---------------- 4. tool-call / agent transcripts in three wire formats (invented repositories and outputs) ----------------
def lines(prefix, n, f): return "\n".join(f(i) for i in range(n))
GO_FILE = "package inventory\n\nimport (\n\t\"errors\"\n\t\"sync\"\n)\n\ntype Item struct {\n\tSKU   string\n\tCount int\n\tPrice float64\n}\n\ntype Store struct {\n\tmu    sync.Mutex\n\titems map[string]*Item\n}\n\nfunc NewStore() *Store { return &Store{items: map[string]*Item{}} }\n\nfunc (s *Store) Add(sku string, n int, price float64) {\n\ts.mu.Lock()\n\tdefer s.mu.Unlock()\n\tif it, ok := s.items[sku]; ok {\n\t\tit.Count += n\n\t\treturn\n\t}\n\ts.items[sku] = &Item{SKU: sku, Count: n, Price: price}\n}\n\nfunc (s *Store) Remove(sku string, n int) error {\n\tit, ok := s.items[sku]\n\tif !ok {\n\t\treturn errors.New(\"unknown sku\")\n\t}\n\tit.Count -= n\n\treturn nil\n}\n\nfunc (s *Store) Total() float64 {\n\tvar t float64\n\tfor _, it := range s.items {\n\t\tt += float64(it.Count) * it.Price\n\t}\n\treturn t\n}\n"
TS_FILE = "export interface Booking { id: string; room: string; start: Date; end: Date; guest: string }\n\nexport class Calendar {\n  private bookings: Booking[] = [];\n\n  add(b: Booking): boolean {\n    for (const x of this.bookings) {\n      if (x.room === b.room && x.start <= b.end && b.start <= x.end) return false;\n    }\n    this.bookings.push(b);\n    return true;\n  }\n\n  cancel(id: string): void {\n    this.bookings = this.bookings.filter(b => b.id != id);\n  }\n\n  forRoom(room: string): Booking[] {\n    return this.bookings.filter(b => b.room === room).sort((a, b) => a.start.getTime() - b.start.getTime());\n  }\n\n  nights(id: string): number {\n    const b = this.bookings.find(b => b.id === id)!;\n    return (b.end.getTime() - b.start.getTime()) / 86400000;\n  }\n}\n"
RS_FILE = "use std::collections::HashMap;\n\n#[derive(Debug, Clone)]\npub struct Reading { pub sensor: String, pub celsius: f64, pub ts: u64 }\n\npub struct Window { size: usize, data: HashMap<String, Vec<Reading>> }\n\nimpl Window {\n    pub fn new(size: usize) -> Self { Window { size, data: HashMap::new() } }\n\n    pub fn push(&mut self, r: Reading) {\n        let v = self.data.entry(r.sensor.clone()).or_default();\n        v.push(r);\n        if v.len() > self.size { v.remove(0); }\n    }\n\n    pub fn mean(&self, sensor: &str) -> f64 {\n        let v = &self.data[sensor];\n        v.iter().map(|r| r.celsius).sum::<f64>() / v.len() as f64\n    }\n\n    pub fn max(&self, sensor: &str) -> Option<f64> {\n        self.data.get(sensor)?.iter().map(|r| r.celsius).fold(None, |m, x| Some(m.map_or(x, |m: f64| m.max(x))))\n    }\n}\n"
TEST_OUT = "$ go test ./...\n--- FAIL: TestRemoveBelowZero (0.00s)\n    store_test.go:41: Remove(\"A-100\", 5) on count 3: expected error, got nil\n--- FAIL: TestConcurrentRemove (0.01s)\n    store_test.go:77: race detected during execution of test\nFAIL\nFAIL\texample.com/inventory\t0.214s\nFAIL"
OAI_TOOLS = '[{"type":"function","function":{"name":"fs_read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},{"type":"function","function":{"name":"fs_write","description":"Write a whole file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},{"type":"function","function":{"name":"search","description":"Search the repository","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]'
def oai(task, steps): return ("You are an autonomous software engineer. Tools are given as an OpenAI function list. To call one, answer with a single line\n"
    "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>\nand nothing else; when the task is finished answer with a summary in prose.\n\nTOOLS = " + OAI_TOOLS +
    "\n\nUSER: " + task + "\n" + "\n".join(steps) + "\nASSISTANT:")
def xmlfmt(task, steps): return ("You operate a terminal on the user's machine. Reply with exactly one action block per turn:\n<action name=\"read\"><path>...</path></action>\n"
    "<action name=\"edit\"><path>...</path><find>...</find><replace>...</replace></action>\n<action name=\"bash\"><cmd>...</cmd></action>\n<action name=\"done\"><report>...</report></action>\n"
    "Think briefly in a <plan> block before the action.\n\nTask: " + task + "\n\n" + "\n\n".join(steps) + "\n\nNext turn:")
def react(task, steps): return ("Answer the question using the tools. Use this format:\nThought: ...\nAction: tool_name[input]\nObservation: (given to you)\n... repeat ...\nFinal Answer: ...\n"
    "Tools: lookup[term] (encyclopedia), calc[expression], weather[city, date], timetable[from, to, date], convert[amount, from_unit, to_unit]\n\nQuestion: " + task + "\n" + "\n".join(steps) + "\nThought:")
AG_FIT = [
 oai("The tests in the inventory package fail. Fix the bugs.", ['ASSISTANT: <tool_call>{"name": "shell", "arguments": {"command": "go test ./..."}}</tool_call>', "TOOL: " + TEST_OUT,
     'ASSISTANT: <tool_call>{"name": "fs_read", "arguments": {"path": "inventory/store.go"}}</tool_call>', "TOOL: " + GO_FILE]),
 oai("Add a method Low(threshold int) []string to the inventory store that returns the SKUs with a count below the threshold, sorted. Write the whole file back.",
     ['ASSISTANT: <tool_call>{"name": "fs_read", "arguments": {"path": "inventory/store.go"}}</tool_call>', "TOOL: " + GO_FILE]),
 oai("Find out why double bookings are still possible for back-to-back stays and fix calendar.ts.", ['ASSISTANT: <tool_call>{"name": "search", "arguments": {"query": "class Calendar"}}</tool_call>',
     "TOOL: src/calendar.ts:3: export class Calendar {", 'ASSISTANT: <tool_call>{"name": "fs_read", "arguments": {"path": "src/calendar.ts"}}</tool_call>', "TOOL: " + TS_FILE]),
 oai("Write unit tests for src/calendar.ts with vitest, covering overlap, cancel and nights. Create the test file.", ['ASSISTANT: <tool_call>{"name": "fs_read", "arguments": {"path": "src/calendar.ts"}}</tool_call>', "TOOL: " + TS_FILE]),
 oai("Our sensor window panics on an unknown sensor. Make mean() return Option<f64> and update the file.", ['ASSISTANT: <tool_call>{"name": "fs_read", "arguments": {"path": "src/window.rs"}}</tool_call>', "TOOL: " + RS_FILE]),
 oai("Summarise what this repository does for a new team member.", ['ASSISTANT: <tool_call>{"name": "shell", "arguments": {"command": "ls -R | head -40"}}</tool_call>',
     "TOOL: .:\nCargo.toml\nREADME.md\nsrc\ntests\n\n./src:\nlib.rs\nwindow.rs\nalerts.rs\nexport.rs\n\n./tests:\nwindow_test.rs\nalerts_test.rs", 'ASSISTANT: <tool_call>{"name": "fs_read", "arguments": {"path": "src/window.rs"}}</tool_call>', "TOOL: " + RS_FILE]),
 xmlfmt("The nightly backup job stopped working yesterday. Find out why and fix it.", ["<plan>Look at the job's log first.</plan>\n<action name=\"bash\"><cmd>tail -20 /var/log/backup/nightly.log</cmd></action>",
     "RESULT:\n2031-03-02 02:00:01 start backup of /srv/data to /mnt/backup/2031-03-02\n2031-03-02 02:00:01 rsync: mkdir \"/mnt/backup/2031-03-02\" failed: No space left on device (28)\n2031-03-02 02:00:01 rsync error: error in file IO (code 11) at main.c(791)\n2031-03-02 02:00:01 backup FAILED rc=11",
     "<plan>Check disk usage on the backup mount.</plan>\n<action name=\"bash\"><cmd>df -h /mnt/backup; ls /mnt/backup | wc -l; ls /mnt/backup | head -5</cmd></action>",
     "RESULT:\nFilesystem      Size  Used Avail Use% Mounted on\n/dev/sdb1       1.8T  1.8T     0 100% /mnt/backup\n412\n2030-01-14\n2030-01-15\n2030-01-16\n2030-01-17\n2030-01-18"]),
 xmlfmt("Rename the configuration key `timeout_ms` to `request_timeout_ms` everywhere in this project and keep reading the old key as a fallback.",
     ["<plan>Find all uses.</plan>\n<action name=\"bash\"><cmd>grep -rn timeout_ms --include=*.py .</cmd></action>",
      "RESULT:\n./app/config.py:14:    timeout_ms: int = 3000\n./app/config.py:38:        self.timeout_ms = int(raw.get(\"timeout_ms\", 3000))\n./app/client.py:22:        self._session.timeout = cfg.timeout_ms / 1000\n./app/client.py:57:            log.warning(\"request exceeded %d ms\", cfg.timeout_ms)\n./tests/test_config.py:9:    assert load({\"timeout_ms\": 50}).timeout_ms == 50"]),
 xmlfmt("Set up a Python virtual environment for this project, install its dependencies and run the test suite. Report what fails.",
     ["<plan>See what dependency files exist.</plan>\n<action name=\"bash\"><cmd>ls; cat pyproject.toml</cmd></action>",
      "RESULT:\nREADME.md  pyproject.toml  src  tests\n[project]\nname = \"harbor\"\nversion = \"0.4.1\"\nrequires-python = \">=3.11\"\ndependencies = [\"httpx>=0.27\", \"pydantic>=2.6\", \"click>=8.1\"]\n\n[project.optional-dependencies]\ndev = [\"pytest>=8\", \"pytest-asyncio\", \"ruff\"]",
      "<plan>Create the environment and install with the dev extras.</plan>\n<action name=\"bash\"><cmd>python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]' && pytest -q</cmd></action>",
      "RESULT:\n...F..F....                                                              [100%]\nFAILED tests/test_cli.py::test_export_csv - AssertionError: assert 'id;name' in 'id,name\\n1,alpha\\n'\nFAILED tests/test_client.py::test_retry_on_503 - RuntimeError: Event loop is closed\n2 failed, 9 passed in 1.42s"]),
 react("How long does it take, door to door, to travel from Zurich main station to Milan Centrale on a weekday morning, and what will the weather be on arrival on 14 May?",
     [" I need the timetable first.", "Action: timetable[Zurich HB, Milano Centrale, 2031-05-14]", "Observation: EC 311 dep 07:33 arr 10:50 direct; EC 313 dep 09:33 arr 12:50 direct; IR + RE dep 08:02 arr 12:35 one change at Chiasso."]),
 react("A recipe needs 2.5 cups of flour and 12 ounces of butter. Give both in grams, and say how much the ingredients weigh together in kilograms.",
     [" Convert the flour first.", "Action: convert[2.5, cup of flour, gram]", "Observation: 2.5 cups of all-purpose flour = 312.5 g"]),
 react("Who designed the lighthouse at Cordouan, when was it finished, and how many years passed between that and the first use of a Fresnel lens there?",
     [" Look up the lighthouse.", "Action: lookup[Cordouan lighthouse]", "Observation: The Cordouan lighthouse stands at the mouth of the Gironde estuary. Designed by Louis de Foix, begun in 1584 and completed in 1611. In 1823 it became the first lighthouse fitted with a Fresnel lens."]),
]
for p in AG_FIT: fit("agent", p, 384)
fit("agent", "You are a customer-support assistant with these functions: get_order(order_id), refund(order_id, amount, reason), escalate(ticket, team). Reply ONLY with JSON: {\"say\": string, \"calls\": [{\"fn\": string, \"args\": object}]}.\n\nConversation so far:\ncustomer: My order 88-1207 arrived with a cracked screen. I want my money back.\nassistant: {\"say\": \"I'm sorry about that. Let me look up the order.\", \"calls\": [{\"fn\": \"get_order\", \"args\": {\"order_id\": \"88-1207\"}}]}\nfunction get_order -> {\"order_id\": \"88-1207\", \"item\": \"Tablet 11 inch\", \"paid\": 329.00, \"currency\": \"EUR\", \"delivered\": \"2031-02-11\", \"status\": \"delivered\"}\ncustomer: Well?\nassistant:")
fit("agent", "Plan, as a numbered list of tool invocations with their exact arguments, how an agent with the tools `sql(query)`, `plot(spec_json)` and `email(to, subject, body)` should answer this request, and write out every query and the final email: \"Send the regional sales leads a chart of last quarter's revenue by region and the three products whose revenue fell most compared with the quarter before. Tables: orders(id, product_id, region, amount, ordered_at), products(id, name, category).\"", 448)
# fresh agent prompts: their own wire format and their own files (nothing pasted here appears in the fitting corpus)
PY_FILE = "import csv\nfrom dataclasses import dataclass\n\n\n@dataclass\nclass Trip:\n    rider: str\n    km: float\n    minutes: int\n    surge: float = 1.0\n\n\nclass FareTable:\n    BASE = 2.5\n    PER_KM = 1.1\n    PER_MIN = 0.3\n\n    def __init__(self):\n        self.trips = []\n\n    def load(self, path):\n        with open(path) as f:\n            for row in csv.DictReader(f):\n                self.trips.append(Trip(row['rider'], float(row['km']), int(row['minutes']), float(row.get('surge', 1))))\n\n    def fare(self, t):\n        return (self.BASE + t.km * self.PER_KM + t.minutes * self.PER_MIN) * t.surge\n\n    def by_rider(self):\n        out = {}\n        for t in self.trips:\n            out[t.rider] = out.get(t.rider, 0) + self.fare(t)\n        return out\n\n    def top(self, n):\n        return sorted(self.by_rider().items(), key=lambda kv: kv[1])[:n]\n"
JAVA_FILE = "package club.members;\n\nimport java.time.LocalDate;\nimport java.util.ArrayList;\nimport java.util.List;\n\npublic class Roster {\n    public static class Member {\n        String name; LocalDate joined; boolean paid;\n        Member(String name, LocalDate joined) { this.name = name; this.joined = joined; }\n    }\n\n    private final List<Member> members = new ArrayList<>();\n\n    public void join(String name, LocalDate when) {\n        members.add(new Member(name, when));\n    }\n\n    public boolean markPaid(String name) {\n        for (Member m : members) {\n            if (m.name == name) { m.paid = true; return true; }\n        }\n        return false;\n    }\n\n    public List<String> unpaidSince(LocalDate cutoff) {\n        List<String> out = new ArrayList<>();\n        for (Member m : members) {\n            if (!m.paid && m.joined.isAfter(cutoff)) out.add(m.name);\n        }\n        return out;\n    }\n\n    public int size() { return members.size(); }\n}\n"
def rpc(task, steps): return ("SYSTEM: You are a maintenance bot. Each turn you emit ONE request object on a single line, in this form:\n"
    "{\"id\": <n>, \"method\": \"open\" | \"save\" | \"exec\" | \"finish\", \"params\": {...}}\nopen takes {\"file\"}, save takes {\"file\", \"text\"}, exec takes {\"argv\": [...]}, finish takes {\"message\"}.\n\nGOAL: " + task + "\n\n" + "\n".join(steps) + "\n>>>")
AG_FRESH = [
 rpc("top(n) in fares.py should return the riders who paid the MOST. Fix it and save the file.", ['>>> {"id": 1, "method": "open", "params": {"file": "fares.py"}}', "<<< " + PY_FILE]),
 rpc("Add a method `average_speed(rider)` in km/h to FareTable and save the whole file.", ['>>> {"id": 1, "method": "open", "params": {"file": "fares.py"}}', "<<< " + PY_FILE]),
 rpc("markPaid never finds anybody when names come from user input. Explain why, fix Roster.java and save it.", ['>>> {"id": 1, "method": "open", "params": {"file": "src/club/members/Roster.java"}}', "<<< " + JAVA_FILE]),
 rpc("The web server returns 502 since this morning. Diagnose it.", ['>>> {"id": 1, "method": "exec", "params": {"argv": ["sh", "-c", "tail -5 /var/log/nginx/error.log; systemctl is-active app.service"]}}',
     "<<< 2031/04/09 07:12:44 [error] 811#811: *9412 connect() failed (111: Connection refused) while connecting to upstream, client: 10.0.3.7, server: shop.example, upstream: \"http://127.0.0.1:8000/\"\nfailed",
     '>>> {"id": 2, "method": "exec", "params": {"argv": ["journalctl", "-u", "app.service", "-n", "8", "--no-pager"]}}',
     "<<< Apr 09 07:12:01 web1 app[20331]: Traceback (most recent call last):\nApr 09 07:12:01 web1 app[20331]:   File \"/srv/app/main.py\", line 12, in <module>\nApr 09 07:12:01 web1 app[20331]:     settings = Settings.load(\"/etc/app/settings.toml\")\nApr 09 07:12:01 web1 app[20331]: tomllib.TOMLDecodeError: Invalid value (at line 31, column 9)\nApr 09 07:12:01 web1 systemd[1]: app.service: Main process exited, code=exited, status=1/FAILURE"]),
 "You have a calculator function calc(expr) and a unit function unit(amount, from, to). Show each call you make and its result on its own line as `call -> result`, then the answer.\n\nHow many litres of paint are needed for two coats on a wall 4.2 m wide and 2.6 m high with a 0.9 m by 2.0 m door, if one litre covers 11 square metres, and how many US gallons is that?",
 "You can query a weather service with forecast(city, date) which returns JSON. Write the two queries you need, invent plausible JSON replies, and then answer: is it warmer in Reykjavik or in Tromso on 21 June, and by how many degrees Fahrenheit?",
 "You are a home-automation agent. Devices: lamp.living, lamp.kitchen, thermostat.hall (setpoint C), blind.bedroom (0-100), lock.front. Commands are JSON lines: {\"device\": ..., \"set\": {...}} or {\"query\": device}. The user says: \"I'm going to bed. Make the house ready for the night, it should be 17 degrees, and tell me if the front door is unlocked.\" Write the JSON lines you send, the replies you would expect, and what you finally tell the user.",
 "Below is the log of an agent that tried to deploy a service and failed. Write a post-mortem: timeline, root cause, what the agent should have checked, and the corrected sequence of commands.\n\n[1] run: docker build -t registry.local/pay:1.9 .  -> ok (41 s)\n[2] run: docker push registry.local/pay:1.9 -> ok\n[3] run: kubectl set image deploy/pay pay=registry.local/pay:1.9 -n prod -> deployment.apps/pay image updated\n[4] run: kubectl rollout status deploy/pay -n prod --timeout=60s -> error: timed out waiting for the condition\n[5] run: kubectl get pods -n prod -l app=pay -> pay-7c9d-x2k 0/1 CrashLoopBackOff 4; pay-5f6b-q8m 1/1 Running 0\n[6] run: kubectl logs pay-7c9d-x2k -n prod -> FATAL: missing required env PAY_LEDGER_URL\n[7] run: kubectl rollout status deploy/pay -n prod --timeout=60s -> error: timed out waiting for the condition\n[8] agent: deployment finished successfully."]
for p in AG_FRESH: fresh("agent", p)

# ---------------- 5. math / science / reasoning ----------------
M_FIT = ["A cylindrical tank of radius 1.2 m and height 3 m is filled at 40 litres a minute. How long until it is full? Show each step.",
 "Prove that the sum of the first n odd numbers is n squared, by induction and then with a picture described in words.",
 "Solve the system 3x + 2y - z = 7, x - y + 4z = 3, 2x + 5y + z = 12 by Gaussian elimination, showing every row operation.",
 "A loan of 18,000 at 6.5 % yearly interest is repaid monthly over 5 years. Derive the payment formula and compute the payment.",
 "A factory's alarm goes off for 97 of 100 real faults and for 5 of 100 fault-free hours; faults happen in 1 hour of 400. When the alarm sounds, how likely is a real fault? Work it out with a table of counts.",
 "Find the eigenvalues and eigenvectors of the matrix [[4, 1], [2, 3]] and diagonalise it.",
 "How many ways can 8 people sit around a round table if two particular people refuse to sit together? Explain.",
 "Differentiate f(x) = x^2 * sin(3x) * e^(-x) and find where the derivative is zero on [0, 2] approximately.",
 "A train leaves at 9:10 at 84 km/h; another leaves the same station at 9:40 on the same track at 120 km/h. When and where does the second catch the first?",
 "Explain why the harmonic series diverges but the sum of 1/n^2 converges. Give the classical arguments.",
 "Compute the probability of getting at least two sixes in five rolls of a fair die. Give an exact fraction and a decimal.",
 "Derive the formula for the volume of a sphere using integration, step by step.",
 "Three resistors of 120, 180 and 360 ohms are in parallel across 24 volts. Find the total resistance, each current and the power.",
 "Balance the combustion of octane and compute the mass of CO2 produced by burning 50 litres of petrol (density 0.75 kg/l).",
 "Explain the difference between mean, median and mode using the data set 3, 7, 7, 8, 12, 13, 41, and say which describes it best.",
 "A ladder 6.5 m long leans against a wall with its foot 2.5 m from the wall. The foot slides away at 0.3 m/s. How fast does the top fall at that moment?",
 "Convert the repeating decimal 0.2(45) to a fraction in lowest terms and explain the method in general.",
 "State and prove the pigeonhole principle, then use it to show that among 13 integers two have the same remainder mod 12.",
 "Using logarithms, estimate how many digits 7^250 has. Show the reasoning.",
 "Explain how the RSA cryptosystem works with the small primes 61 and 53: generate keys, encrypt the message 65, decrypt it.",
 "Five friends finish a race. Ana is ahead of Ben, Cy is behind Dee, Ben is ahead of Dee, Eli is not last and not first, Cy is ahead of Eli. Find every possible order and explain.",
 "Describe the two-slit experiment and what it implies, at the level of a first-year physics course, with the relevant formula.",
 "Estimate, with stated assumptions, how many piano tuners work in a city of two million people.",
 "Explain big-O, big-Theta and big-Omega with three example functions each, and rank n log n, n^1.5, 2^n, n!, log^2 n."]
for p in M_FIT: fit("math", p)
M_FRESH = ["A rectangular garden is 4 m longer than it is wide and has an area of 96 square metres. Find its sides and the length of fencing needed.",
 "Integrate x * ln(x) from 1 to e, showing the integration by parts.",
 "In how many ways can the letters of the word BANANA be arranged? And with the two N's not adjacent?",
 "Solve the recurrence a(n) = 5a(n-1) - 6a(n-2) with a(0) = 2, a(1) = 5 and verify the first five terms.",
 "From a drawer holding 5 red, 4 green and 3 blue socks you pull out three at once in the dark. How likely is it that you hold one sock of each colour?",
 "Find the shortest distance from the point (3, -1, 2) to the plane 2x - y + 2z = 4 and the closest point on the plane.",
 "A 2 kg block slides down a 30 degree incline 5 m long with friction coefficient 0.2. Find its speed at the bottom.",
 "Show that log base 2 of 3 cannot be written as a fraction of two integers.",
 "What is the remainder when 3^200 is divided by 13? Explain using Fermat's little theorem.",
 "A company's revenue grows 12 % a year. How many years until it triples? Solve exactly and give a decimal.",
 "Explain the Monty Hall problem and compute the winning probabilities for switching and staying, also for the variant with five doors where the host opens three.",
 "Find the area between the curves y = x^2 and y = 2x + 3.",
 "The half-life of a substance is 14 days. How much of a 200 mg sample remains after 60 days, and when does it fall below 1 mg?",
 "Two dice are rolled. Given that the sum is at least 9, what is the probability that at least one die shows a 6?",
 "Compute the determinant and the inverse of [[2, 0, 1], [1, 3, 2], [1, 1, 1]].",
 "A cone of height 12 cm and base radius 5 cm is cut parallel to the base 4 cm from the top. Find the volume of the remaining frustum."]
for p in M_FRESH: fresh("math", p)

# ---------------- 6. structured output ----------------
S_FIT = ["Produce a JSON array of 12 fictional employees with fields id, name, department, hire_date (ISO), salary and skills (array). Output only JSON.",
 "Write a CSV file (with header) of 25 rows of fictional weather observations: station, date, temp_min, temp_max, rain_mm, wind_kph.",
 "Create a Markdown table comparing eight programming languages by typing discipline, memory management, typical use and year of first release.",
 "Write a YAML configuration for a fictional data pipeline with sources, transforms, sinks, schedules and retry policies, with comments.",
 "Write an XML document describing a library catalogue with ten books, each with nested authors, publisher and availability, plus a DTD.",
 "Give a week-long meal plan as a table: day, breakfast, lunch, dinner, approximate calories.",
 "Write a JSON Schema (draft 2020-12) for an invoice with line items, taxes and a discriminated payment method.",
 "Produce a changelog in Keep-a-Changelog format for six fictional releases of a command-line tool.",
 "Write an iCalendar (.ics) file with five events including a weekly recurring meeting with exceptions.",
 "List the 30 largest moons of the solar system as lines of the form `name | planet | radius km`, as far as you know them.",
 "Write a TOML file for a game's settings (graphics, audio, controls with key bindings, network) and a matching INI version.",
 "Produce a nested bullet outline, four levels deep, for a textbook on operating systems.",
 "Write SQL INSERT statements for 20 rows into a table orders(id, customer, product, qty, unit_price, ordered_at).",
 "Write a unified diff that renames a function and updates three call sites in a small invented C file.",
 "Count down from 150 to 1 in steps of 3, comma separated, then give the sum.",
 "Write a regular grammar in EBNF for a small configuration language with sections, typed values, lists and comments."]
for p in S_FIT: fit("struct", p)
for p in ["Describe a made-up music album as one JSON document: title, artist, year and twelve tracks, each with its title, length in seconds and writers. No prose around it.",
 "Write a Markdown table of the planets with columns: name, mean distance from the Sun (AU), orbital period, number of known moons, one notable fact.",
 "Write a CSV (with header) of 20 fictional bank transactions: date, account, counterparty, amount, currency, category.",
 "Write a YAML file describing a CI pipeline with stages lint, test (matrix of three versions), build and deploy with manual approval.",
 "List the numbers from 1 to 120 that are divisible by 3 or 7, space separated, and nothing else.",
 "Write an HTML table of a fictional football league: 10 teams, played, won, drawn, lost, goals for, goals against, points.",
 "Give a packing list for a two-week winter trip as nested JSON by category with quantities.",
 "Write a crontab file with ten entries for a small server, each with a comment explaining its schedule."]: fresh("struct", p)

# ---------------- 7. pasted logs / configs / data to analyse (invented, generated) ----------------
def access_log(n, seed):
    paths = ["/", "/login", "/api/v2/cart", "/api/v2/items?page=%d", "/static/app.js", "/checkout", "/api/v2/search?q=lamp", "/healthz"]; st = [200, 200, 200, 304, 200, 500, 200, 404, 200, 502]
    return "\n".join(f"10.4.{(i * 7 + seed) % 250}.{(i * 13 + 5) % 250} - - [17/Mar/2031:14:{(i // 4) % 60:02d}:{(i * 17) % 60:02d} +0000] \"{'POST' if i % 5 == 1 else 'GET'} {paths[(i + seed) % 8] % ((i,) if '%d' in paths[(i + seed) % 8] else ())} HTTP/1.1\" {st[(i * 3 + seed) % 10]} {(i * 977 + seed * 31) % 9000 + 120} {((i * 37 + seed) % 900) / 1000 + 0.004:.3f}" for i in range(n))
def csv_sales(n, seed):
    reg = ["north", "south", "east", "west", "export"]; pr = ["kettle", "toaster", "lamp", "fan", "heater", "blender"]
    return "date,region,product,units,unit_price\n" + "\n".join(f"2031-{1 + (i + seed) % 12:02d}-{1 + (i * 3) % 28:02d},{reg[(i + seed) % 5]},{pr[(i * 5 + seed) % 6]},{(i * 7 + seed) % 40 + 1},{19.9 + ((i * 11 + seed) % 9) * 5:.2f}" for i in range(n))
fit("data", "Here is an excerpt of a web server access log (last field is the response time in seconds). Summarise the traffic, point out the failing endpoints and suggest what to investigate.\n\n" + access_log(40, 3), 384)
fit("data", "Analyse this sales extract: total units and revenue by region and by product, the best month, and anything odd. Show your working as a table.\n\n" + csv_sales(36, 2), 448)
fit("data", "Convert the following CSV into a JSON array of objects, with units as an integer and unit_price as a number. Output only JSON.\n\n" + csv_sales(14, 9), 512)
fit("data", "Rewrite this access log as a Markdown table with columns ip, time, method, path, status, bytes, seconds. Keep every row.\n\n" + access_log(12, 5), 512)
fit("data", "Review this configuration file for mistakes and security problems, then give a corrected version in full.\n\n[server]\nlisten = 0.0.0.0:8080\nworkers = 0\ntls = off\nadmin_password = admin\n\n[database]\nurl = postgres://app:app@db.internal:5432/app?sslmode=disable\npool_size = 500\ntimeout = 0\n\n[cache]\nbackend = redis\nurl = redis://cache.internal\nttl_seconds = -1\n\n[logging]\nlevel = trace\nfile = /tmp/app.log\nrotate = never\n\n[uploads]\nmax_size_mb = 100000\nallowed = *\npath = ../uploads", 448)
fit("data", "Explain what this stack trace means, the most likely cause, and how to fix it.\n\nException in thread \"pool-3-thread-7\" java.util.ConcurrentModificationException\n\tat java.base/java.util.ArrayList$Itr.checkForComodification(ArrayList.java:1095)\n\tat java.base/java.util.ArrayList$Itr.next(ArrayList.java:1049)\n\tat com.example.billing.InvoiceBatch.total(InvoiceBatch.java:88)\n\tat com.example.billing.Nightly.lambda$run$2(Nightly.java:41)\n\tat java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)\n\tat java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)\n\tat java.base/java.lang.Thread.run(Thread.java:1583)")
fresh("struct", "Convert this list into a Markdown table with a totals row.\n\n" + "\n".join(f"crate {chr(65 + i)}: {12 + i * 7 % 23} boxes of {['pears', 'plums', 'figs', 'limes'][i % 4]} at {3.4 + (i * 13 % 7) * 0.45:.2f} each, bay {1 + i * 5 % 9}" for i in range(12)))

# ---------------- 8. rewrite tasks on files written here (copy-heavy output, other files than the bench's edit set) ----------------
fit("edit", "Here is the file store.go:\n\n```go\n" + GO_FILE + "```\n\nRewrite the whole file so that every method takes the mutex, Remove refuses to go below zero, and every exported symbol has a doc comment. Output the complete file.", 512)
fit("edit", "Here is the file calendar.ts:\n\n```ts\n" + TS_FILE + "```\n\nRewrite the whole file with strict equality everywhere, readonly fields in the interface and an explicit error instead of the non-null assertion. Output the complete file.", 512)
fit("edit", "Here is the file window.rs:\n\n```rust\n" + RS_FILE + "```\n\nAdd a method `min` next to `max`, derive PartialEq on Reading, and add rustdoc comments on every public item. Output the complete file.", 512)
fit("edit", "Translate this Go file to Python, keeping names and behaviour, with type hints and a threading.Lock:\n\n```go\n" + GO_FILE + "```", 512)
fit("edit", "Port this TypeScript file to Kotlin, idiomatically, keeping the public API:\n\n```ts\n" + TS_FILE + "```", 512)
fit("edit", "Proofread and lightly edit the following text. Keep the author's wording wherever it is correct and output the full corrected text.\n\nThe harbour was more quiet then usual that morning. Fishermens boats, which normaly left before the sun rise, was still tied to the pier, there nets piled in heaps that smelled of salt and diesel. Nobody could of said exactly what was wrong, but the gulls who usually fighted over scraps were sitting silent on the breakwater. Old Marta, who has sold coffee from the same kiosk for fourty years, looked at the sky and than at the water and begun to close her shutters. \"When the birds dont eat,\" she said to noone in particular, \"its time to go home.\" By noon the wind had rised, and the first waves were braking over the outer wall.", 448)
fresh("code", "Here is the file fares.py:\n\n```python\n" + PY_FILE + "```\n\nRewrite the whole file with type hints on every method, a `__len__`, and `load` skipping rows with a negative distance. Output the complete file.")
fresh("code", "Here is the file Roster.java:\n\n```java\n" + JAVA_FILE + "```\n\nAdd a `leave(String name)` method and make `unpaidSince` return the names sorted. Output the complete file.")

# ---------------- 9. long-form (more tokens a request) ----------------
LONG = ["Write the first chapter (about 900 words) of a mystery novel set in a mountain observatory during a snowstorm.",
 "Write a complete technical design document for a URL-shortening service: goals, API, data model, capacity estimates, failure modes, rollout.",
 "Write a detailed travel guide to an imaginary island nation: geography, history, cuisine, customs, a seven-day itinerary.",
 "Write a complete single-file Python program (about 200 lines) that implements a text adventure game with rooms, items, a parser and save / load.",
 "Write a long-form explainer on how the electrical grid keeps supply and demand in balance, covering frequency, reserves, markets and storage.",
 "Write a complete C header and implementation of a dynamic array of generic elements with growth, insertion, removal, sort and binary search, with tests.",
 "Write a speech of about 800 words for the opening of a new public library, then a 100-word summary of it.",
 "Write a detailed comparison of relational, document, key-value, column-family and graph databases with examples and guidance on choosing."]
for p in LONG: fit("long", p, 1024)

# ---------------- 10. the same kinds with thinking on (greedy; reasoning text is drafted too) ----------------
TH = ["A farmer has 100 m of fence and wants the largest rectangular pen against a straight river (no fence on the river side). What are the dimensions?",
 "Write a function in Rust that returns the k-th smallest element of two sorted slices in logarithmic time, and explain why it is correct.",
 "Which is larger, 2^100 or 3^63? Decide without a calculator and justify.",
 "Plan a database migration from a single Postgres server to a primary with two replicas with less than a minute of downtime.",
 "Twelve coins, one is counterfeit and either heavier or lighter. Find it in three weighings on a balance scale and say whether it is heavy or light.",
 "Explique pourquoi le ciel est bleu le jour et rouge au coucher du soleil.", "请分析为什么短视频平台容易让人上瘾，并提出三条应对建议。",
 "Design a fair schedule for six teams playing each other once over five weekends with three pitches.",
 "Review this function for bugs: def mid(a, b): return (a + b) / 2 if a < b else mid(b, a) -- used for binary search on integer indexes in Python and translated to C with int.",
 "Is it cheaper to lease a car at 310 a month for four years with 2,500 down or to buy it for 24,000 with a resale value of 11,000? State assumptions.",
 "Write a haiku sequence of five haiku about a river through the four seasons and back to spring.",
 "A tool returned this JSON: {\"rows\": [[\"A\", 3, 4.5], [\"B\", 0, 9.0], [\"C\", 7, 1.25]], \"cols\": [\"sku\", \"qty\", \"price\"]}. Compute the stock value per SKU and in total and answer as a Markdown table."]
for p in TH: fit("think", p, 448, True)

json.dump({}, open(os.devnull, "w"))
for name, S in (("fit", FIT), ("fresh", FRESH)):
    with open(os.path.join(out, name + ".jsonl"), "w") as f:
        for x in S: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    c = {}
    for x in S: c[x["cat"].split("-")[0]] = c.get(x["cat"].split("-")[0], 0) + 1
    print(name, len(S), "prompts", c, "token budget", sum(x.get("max_tokens", 384) for x in S))
