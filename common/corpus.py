"""A small, fully fictional interlinked knowledge base + a labeled eval set.

Fictional on purpose: the answers can't be in any model's training data, so a method only scores
well if it actually *retrieves* the right documents — this measures RAG, not memorization. The
docs cross-reference each other (people → companies → products → dependencies) so multi-hop and
graph methods have something real to chain. Every question records the gold relevant doc ids and a
reference answer, so retrieval recall and answer correctness are both measurable.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Doc:
    id: str
    title: str
    text: str


@dataclass
class Question:
    qid: str
    question: str
    gold_docs: list[str]          # doc ids that must be retrieved to answer
    answer: str                   # reference answer
    hops: int                     # 1 = single doc, 2+ = needs chaining


DOCS = [
    Doc("d1", "Veyra Systems", "Veyra Systems is a software company founded in 2014 by Mara Lindqvist in Tallinn. Veyra builds developer infrastructure and is best known for its product Quorrel."),
    Doc("d2", "Quorrel", "Quorrel is a stream-processing engine built by Veyra Systems. Quorrel stores its state in the Talix database and is written in the Orsa programming language."),
    Doc("d3", "Talix database", "Talix is a distributed key-value database created by the company Brightfen. Talix is used by several stream processors, including Quorrel, for durable state storage."),
    Doc("d4", "Brightfen", "Brightfen is a database company founded in 2011 by Idris Okonkwo. Its flagship product is the Talix database. Brightfen was acquired by Northwind Capital in 2020."),
    Doc("d5", "Orsa language", "Orsa is a statically typed programming language designed by Petra Voss. Orsa compiles to native code and is used to build Quorrel."),
    Doc("d6", "Mara Lindqvist", "Mara Lindqvist is an Estonian engineer who founded Veyra Systems in 2014. Before Veyra she worked at Brightfen on the Talix database."),
    Doc("d7", "Northwind Capital", "Northwind Capital is a private equity firm. It acquired Brightfen in 2020 and also owns a stake in the analytics company Pell Metrics."),
    Doc("d8", "Pell Metrics", "Pell Metrics is an analytics company founded by Sun Park. Pell Metrics uses Quorrel to process event streams in real time."),
    Doc("d9", "Idris Okonkwo", "Idris Okonkwo is a database researcher who founded Brightfen in 2011. He previously published the Cascade consistency protocol."),
    Doc("d10", "Cascade protocol", "The Cascade consistency protocol is a method for distributed consensus published by Idris Okonkwo. Talix implements Cascade for replication."),
    Doc("d11", "Petra Voss", "Petra Voss is a language designer who created the Orsa programming language. She currently advises Veyra Systems on compiler design."),
    Doc("d12", "Sun Park", "Sun Park is the founder of Pell Metrics, an analytics company that runs on Quorrel. Park previously worked as a data engineer at Northwind Capital."),
    Doc("d13", "Tallinn tech scene", "Tallinn, Estonia hosts several infrastructure companies. Veyra Systems is headquartered there. The city is known for a strong base of compiler and database engineers."),
    Doc("d14", "Quorrel 3.0 release", "Quorrel 3.0 added exactly-once processing. The release notes credit Petra Voss for compiler improvements to Orsa that reduced Quorrel's memory use by 30%."),
]

QUESTIONS = [
    Question("q1", "Who founded Veyra Systems?", ["d1"], "Mara Lindqvist.", 1),
    Question("q2", "What database does Quorrel use for state storage?", ["d2"], "The Talix database.", 1),
    Question("q3", "What programming language is Quorrel written in?", ["d2"], "Orsa.", 1),
    Question("q4", "Who designed the Orsa language?", ["d5"], "Petra Voss.", 1),
    # multi-hop
    Question("q5", "Who founded the company that makes the database Quorrel uses for state?",
             ["d2", "d3", "d4"], "Idris Okonkwo (founder of Brightfen, which makes Talix).", 3),
    Question("q6", "Which private equity firm owns the company behind the Talix database?",
             ["d3", "d4", "d7"], "Northwind Capital (it acquired Brightfen).", 3),
    Question("q7", "Before founding Veyra, where did Mara Lindqvist work?",
             ["d6"], "At Brightfen, on the Talix database.", 1),
    Question("q8", "Who designed the language used to build the stream processor that Pell Metrics runs on?",
             ["d8", "d2", "d5"], "Petra Voss (Pell Metrics runs on Quorrel, which is written in Orsa, designed by Petra Voss).", 3),
    Question("q9", "What consistency protocol does Talix use for replication, and who published it?",
             ["d10"], "The Cascade protocol, published by Idris Okonkwo.", 1),
    Question("q10", "Name two companies connected to Northwind Capital.",
             ["d7", "d4", "d8"], "Brightfen (acquired) and Pell Metrics (a stake).", 2),
    Question("q11", "Who improved Orsa's compiler for the Quorrel 3.0 release?",
             ["d14"], "Petra Voss.", 1),
    Question("q12", "Which founder previously worked at Northwind Capital?",
             ["d12"], "Sun Park, founder of Pell Metrics.", 1),
]


def docs() -> list[Doc]:
    return list(DOCS)


def questions() -> list[Question]:
    return list(QUESTIONS)
