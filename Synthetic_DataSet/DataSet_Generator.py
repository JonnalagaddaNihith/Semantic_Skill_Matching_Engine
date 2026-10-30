import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
VERSION = "1.0.0"

DOMAINS = {
    "Programming": [
        "Python", "Java", "JavaScript", "C", "C++", "C#", "Go", "Rust",
        "TypeScript", "Kotlin", "PHP", "Ruby", "Swift", "Scala",
        "R programming", "MATLAB", "Perl", "Elixir", "Haskell", "Dart",
        "Bash scripting", "PowerShell",
        "Object-oriented programming", "Data structures", "Algorithms",
        "Design patterns",
    ],
    "Data and AI": [
        "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
        "Cassandra", "DynamoDB", "SQLite",
        "Data analysis", "Data visualization", "Statistics",
        "Machine learning", "Deep learning", "Natural language processing",
        "Computer vision", "Reinforcement learning", "Generative AI", "Large language models",
        "Prompt engineering", "MLOps", "Feature engineering", "Model evaluation",
        "Time series analysis", "Pandas", "NumPy", "scikit-learn",
        "PyTorch", "TensorFlow", "Keras", "Hugging Face",
    ],
    "Web Development": [
        "HTML", "CSS", "Tailwind CSS", "React", "Next.js", "Angular", "Vue.js", "Svelte",
        "Redux", "Node.js", "Express.js", "Flask", "Django", "FastAPI",
        "Spring Boot", ".NET", "Ruby on Rails", "Laravel",
        "REST API development", "GraphQL", "gRPC", "WebSockets", "API design",
        "Web accessibility", "Responsive design", "Progressive Web Apps",
        "Frontend development", "Backend development",
        "Full-stack development", "Authentication", "Web security",
    ],
    "Cloud and DevOps": [
        "Docker", "Kubernetes", "Git", "GitHub Actions", "Jenkins", "GitLab CI",
        "CI/CD", "Linux", "AWS", "Microsoft Azure", "Google Cloud",
        "Terraform", "Ansible", "Monitoring", "Logging",
        "Infrastructure as code", "Cloud architecture",
        "Serverless computing", "Nginx", "Networking", "Deployment",
    ],
    "Software Engineering": [
        "Unit testing", "Integration testing", "Automated testing", "Debugging",
        "Code review", "Agile development", "Scrum",
        "System design", "Microservices", "Software architecture",
        "Requirements analysis", "Technical documentation",
        "Performance optimization", "Concurrency",
        "Secure coding", "Version control", "Refactoring",
        "Clean code",
    ],
    "Security": [
        "Cybersecurity", "Penetration testing", "Application security",
        "Cryptography", "OAuth", "JWT",
    ],
    "Design and Product": [
        "UI design", "UX design", "Figma", "Wireframing", "Prototyping",
        "Product management", "User research",
    ],
    "Business and Analytics": [
        "Excel", "Tableau", "Power BI",
    ],
}

LABEL_SCORES = {
    "UNRELATED": 0,
    "PREREQUISITE": 1,
    "RELATED": 2,
    "PARTIAL_MATCH": 3,
    "PARAPHRASE": 4,
    "EXACT": 4,
}

SKILL_ALIASES = {
    # ===== Programming languages =====
    "Python": ["Python programming", "writing Python code", "Python development", "Python scripting"],
    "Java": ["Java programming", "Java software development", "Java development", "JVM programming"],
    "JavaScript": ["JS programming", "JavaScript development", "ECMAScript", "JavaScript coding"],
    "TypeScript": ["TS programming", "TypeScript development", "typed JavaScript"],
    "C": ["C programming", "C language development", "ANSI C"],
    "C++": ["C++ programming", "C plus plus development", "CPP programming", "modern C++"],
    "C#": ["C sharp programming", "C# development", ".NET C#", "Csharp"],
    "Go": ["Golang", "Go programming", "Go language development"],
    "Rust": ["Rust programming", "Rust systems programming", "Rust development"],
    "Ruby": ["Ruby programming", "Ruby development", "Ruby scripting"],
    "PHP": ["PHP programming", "PHP web development", "PHP scripting"],
    "Swift": ["Swift programming", "Swift iOS development", "Swift development"],
    "Kotlin": ["Kotlin programming", "Kotlin Android development", "Kotlin development"],
    "Scala": ["Scala programming", "Scala development", "functional Scala"],
    "R programming": ["R statistical programming", "R language", "R scripting"],
    "MATLAB": ["MATLAB programming", "MATLAB scripting", "matrix laboratory"],
    "Perl": ["Perl scripting", "Perl programming", "Perl development"],
    "Elixir": ["Elixir programming", "Elixir development", "Elixir/Erlang"],
    "Haskell": ["Haskell programming", "functional Haskell", "Haskell development"],
    "Dart": ["Dart programming", "Dart language", "Flutter Dart"],
    "Bash scripting": ["shell scripting", "Bash scripting", "Unix shell"],
    "PowerShell": ["PowerShell scripting", "Windows PowerShell", "PS scripting"],
    "Object-oriented programming": ["OOP", "object oriented design", "OOP principles"],
    "Data structures": ["data structures", "data organisation", "structuring data"],
    "Algorithms": ["algorithm design", "algorithmic problem solving", "algorithms"],
    "Design patterns": ["software design patterns", "GoF patterns", "architectural patterns"],

    # ===== Web / frontend =====
    "HTML": ["HTML5", "hypertext markup language", "semantic HTML"],
    "CSS": ["CSS3", "cascading style sheets", "styling web pages"],
    "React": ["React.js", "ReactJS", "React user interfaces", "React development"],
    "Next.js": ["NextJS", "Next.js framework", "React with Next.js"],
    "Vue.js": ["Vue", "VueJS", "Vue framework", "Vue development"],
    "Angular": ["AngularJS", "Angular framework", "Angular development"],
    "Svelte": ["Svelte framework", "SvelteJS", "Svelte development"],
    "Redux": ["Redux state management", "Redux store", "Redux toolkit"],
    "Tailwind CSS": ["Tailwind", "TailwindCSS", "utility-first CSS"],
    "Web accessibility": ["a11y", "WCAG compliance", "accessible web design"],
    "Responsive design": ["mobile-friendly interfaces", "adaptive web layouts", "responsive web design", "mobile-first design"],
    "Progressive Web Apps": ["PWA", "PWA development", "installable web apps"],
    "Frontend development": ["frontend engineering", "client-side development", "UI development"],
    "Backend development": ["backend engineering", "server-side development", "server development"],
    "Full-stack development": ["fullstack development", "end-to-end web development", "full stack engineering"],

    # ===== Backend / APIs =====
    "Node.js": ["NodeJS", "Node development", "Node.js runtime", "server-side JavaScript"],
    "Express.js": ["Express", "ExpressJS", "Express framework"],
    "Django": ["Django framework", "Django web development", "Python Django"],
    "Flask": ["Flask framework", "Flask web development", "Python Flask"],
    "FastAPI": ["FastAPI framework", "Python FastAPI", "Fast API development"],
    "Spring Boot": ["SpringBoot", "Spring framework", "Java Spring Boot"],
    ".NET": ["dotnet", ".NET framework", ".NET Core", "ASP.NET"],
    "Ruby on Rails": ["Rails", "RoR", "Rails framework"],
    "Laravel": ["Laravel framework", "PHP Laravel", "Laravel development"],
    "GraphQL": ["GraphQL API", "GraphQL schema", "GraphQL queries"],
    "REST API development": ["RESTful APIs", "HTTP API development", "REST services", "RESTful web services"],
    "gRPC": ["gRPC services", "Google RPC", "gRPC API development"],
    "Microservices": ["microservice architecture", "microservices design", "service-oriented architecture"],
    "WebSockets": ["WebSocket communication", "real-time WebSockets", "socket programming"],
    "API design": ["API architecture", "designing APIs", "API specification"],
    "Authentication": ["user authentication", "login systems", "auth"],
    "Web security": ["web application security", "securing web apps", "OWASP practices"],

    # ===== Data / databases =====
    "SQL": ["structured query language", "relational database querying", "SQL querying"],
    "PostgreSQL": ["Postgres database", "PostgreSQL database management", "Postgres", "PostgreSQL admin"],
    "MySQL": ["MySQL database", "MySQL management", "MySQL server"],
    "MongoDB": ["Mongo database", "MongoDB NoSQL", "document database MongoDB"],
    "Redis": ["Redis cache", "Redis in-memory store", "Redis key-value store"],
    "Elasticsearch": ["Elastic search", "Elasticsearch engine", "ELK stack search"],
    "Cassandra": ["Apache Cassandra", "Cassandra NoSQL", "Cassandra database"],
    "DynamoDB": ["AWS DynamoDB", "Dynamo database", "Amazon DynamoDB"],
    "SQLite": ["SQLite database", "embedded SQL database", "SQLite storage"],
    "Database design": ["schema design", "database modeling", "relational modeling", "data modeling"],

    # ===== ML / AI =====
    "Machine learning": ["ML engineering", "machine learning techniques", "ML development", "machine learning modeling"],
    "Deep learning": ["neural network development", "deep neural networks", "deep learning models", "DL"],
    "Natural language processing": ["NLP", "language processing with AI", "text processing", "computational linguistics"],
    "Computer vision": ["image understanding", "visual AI systems", "CV", "image processing AI"],
    "Reinforcement learning": ["RL", "reward-based learning", "reinforcement learning agents"],
    "Generative AI": ["GenAI", "generative models", "generative artificial intelligence"],
    "Large language models": ["LLMs", "LLM engineering", "large language model development", "foundation models"],
    "Prompt engineering": ["prompt design", "LLM prompting", "prompt optimization"],
    "MLOps": ["ML operations", "machine learning operations", "model deployment pipelines"],
    "Data analysis": ["analysing datasets", "data analytics", "exploratory data analysis", "EDA"],
    "Data visualization": ["visual analytics", "communicating data visually", "dataviz", "charting data"],
    "Statistics": ["statistical analysis", "applied statistics", "statistical modeling"],
    "Feature engineering": ["feature extraction", "feature selection", "feature construction"],
    "Model evaluation": ["model validation", "performance evaluation", "model metrics"],
    "Time series analysis": ["time-series forecasting", "temporal data analysis", "forecasting models"],
    "Pandas": ["pandas library", "Python pandas", "dataframes with pandas"],
    "NumPy": ["NumPy library", "numerical Python", "numpy arrays"],
    "scikit-learn": ["sklearn", "scikit learn", "Python ML library"],
    "TensorFlow": ["TF", "TensorFlow framework", "Google TensorFlow"],
    "PyTorch": ["Torch", "PyTorch framework", "Facebook PyTorch"],
    "Keras": ["Keras API", "Keras deep learning", "Keras models"],
    "Hugging Face": ["HuggingFace", "transformers library", "Hugging Face models"],

    # ===== Cloud / DevOps =====
    "AWS": ["Amazon Web Services", "AWS cloud", "Amazon cloud"],
    "Microsoft Azure": ["Azure", "Azure cloud", "Azure services"],
    "Google Cloud": ["GCP", "Google Cloud Platform", "GCloud"],
    "Docker": ["containerization with Docker", "Docker containers", "Docker images", "Dockerfile"],
    "Kubernetes": ["K8s orchestration", "container orchestration", "K8s", "Kubernetes clusters"],
    "CI/CD": ["continuous integration and delivery", "automated build pipelines", "continuous deployment", "CI CD pipelines"],
    "Jenkins": ["Jenkins CI", "Jenkins pipelines", "Jenkins automation"],
    "GitHub Actions": ["GH Actions", "GitHub CI", "GitHub workflows"],
    "GitLab CI": ["GitLab pipelines", "GitLab CI/CD", "GitLab automation"],
    "Terraform": ["Terraform IaC", "infrastructure as code with Terraform", "HashiCorp Terraform"],
    "Ansible": ["Ansible automation", "Ansible playbooks", "configuration management"],
    "Linux": ["Linux administration", "Linux systems", "Unix/Linux", "Linux command line"],
    "Nginx": ["Nginx server", "Nginx web server", "Nginx reverse proxy"],
    "Serverless computing": ["serverless architecture", "FaaS", "function as a service"],
    "Infrastructure as code": ["IaC", "infrastructure automation", "declarative infrastructure"],
    "Monitoring": ["observability", "system monitoring", "application monitoring"],
    "Logging": ["log aggregation", "centralized logging", "log analysis"],
    "Cloud architecture": ["cloud solution design", "cloud infrastructure design", "cloud systems architecture"],
    "Networking": ["network engineering", "network design", "network configuration"],
    "Deployment": ["deployment automation", "release management", "deploying software"],

    # ===== Software engineering =====
    "Git": ["Git version control", "source code versioning", "Git SCM", "version control with Git"],
    "Version control": ["VC", "SCM", "version control systems"],
    "Unit testing": ["testing individual software components", "unit tests", "component testing"],
    "Integration testing": ["integration tests", "end-to-end integration testing", "system integration testing"],
    "Automated testing": ["test automation", "automated testing frameworks", "automating tests"],
    "Debugging": ["troubleshooting code", "debugging software", "bug fixing"],
    "Code review": ["peer review", "code reviewing", "reviewing pull requests"],
    "Refactoring": ["code refactoring", "improving code quality", "code cleanup"],
    "Agile development": ["Agile methodology", "Agile development", "Agile practices"],
    "Scrum": ["Scrum framework", "Scrum ceremonies", "Agile Scrum"],
    "System design": ["software system architecture", "designing scalable systems", "systems architecture", "distributed system design"],
    "Software architecture": ["architecting software", "software architecture design", "architectural design"],
    "Requirements analysis": ["requirements engineering", "gathering requirements", "requirements elicitation"],
    "Technical documentation": ["tech writing", "documentation writing", "writing technical docs"],
    "Performance optimization": ["performance tuning", "optimising performance", "perf optimisation"],
    "Concurrency": ["concurrent programming", "parallel programming", "concurrent systems"],
    "Secure coding": ["secure programming", "security-conscious coding", "defensive coding"],
    "Clean code": ["readable code", "maintainable code", "clean coding practices"],

    # ===== Security =====
    "Cybersecurity": ["security engineering", "information security", "infosec", "cyber security"],
    "Penetration testing": ["pen testing", "ethical hacking", "security testing"],
    "Application security": ["AppSec", "secure coding", "application security practices"],
    "Cryptography": ["encryption", "cryptographic systems", "crypto engineering"],
    "OAuth": ["OAuth2", "OAuth authentication", "OAuth flows"],
    "JWT": ["JSON Web Tokens", "JWT authentication", "token-based auth"],

    # ===== Design / product =====
    "UI design": ["user interface design", "interface design", "UI/UX design"],
    "UX design": ["user experience design", "UX research", "user experience"],
    "Figma": ["Figma design", "Figma prototyping", "Figma tool"],
    "Wireframing": ["wireframes", "low-fidelity design", "wireframe creation"],
    "Prototyping": ["interactive prototypes", "prototype design", "rapid prototyping"],
    "Product management": ["product ownership", "PM", "product strategy"],
    "User research": ["UX research", "user interviews", "usability research"],

    # ===== Business / analytics =====
    "Excel": ["Microsoft Excel", "spreadsheet modeling", "Excel analysis"],
    "Tableau": ["Tableau dashboards", "Tableau visualization", "Tableau BI"],
    "Power BI": ["PowerBI", "Microsoft Power BI", "Power BI dashboards"],
}

SKILL_TO_DOMAIN = {s: d for d, ss in DOMAINS.items() for s in ss}


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def alias_coverage():
    covered, missing = [], []
    for domain, skills in DOMAINS.items():
        for skill in skills:
            (covered if skill in SKILL_ALIASES else missing).append(skill)
    return {
        "total_skills": len(covered) + len(missing),
        "covered": len(covered),
        "missing": len(missing),
        "coverage_pct": round(100.0 * len(covered) / max(1, len(covered) + len(missing)), 2),
        "missing_skills": sorted(missing),
        "aliases_without_skills": sorted(set(SKILL_ALIASES) - set(SKILL_TO_DOMAIN)),
    }


def make_catalog():
    competencies = []
    index = 1
    for domain, skills in DOMAINS.items():
        for skill in skills:
            aliases = SKILL_ALIASES.get(
                skill,
                [f"{skill} fundamentals", f"practical {skill} skills"],
            )
            competencies.append({
                "competency_id": f"C{index:04d}",
                "competency_name": skill,
                "domain": domain,
                "description": (
                    f"Ability to apply {skill} concepts and techniques "
                    f"to practical {domain.lower()} tasks."
                ),
                "aliases": " | ".join(aliases),
            })
            index += 1
    return competencies


def build_entities(rng, competencies, learner_count=500):
    entities = {
        "learners": [],
        "learner_skills": [],
        "jobs": [],
        "job_requirements": [],
        "courses": [],
        "course_competencies": [],
    }

    for i in range(1, learner_count + 1):
        entities["learners"].append({
            "learner_id": f"L{i:04d}",
            "experience_level": rng.choice(
                ["beginner", "intermediate", "advanced"]
            ),
            "target_domain": rng.choice(list(DOMAINS)),
        })
        for item in rng.sample(competencies, rng.randint(3, 8)):
            entities["learner_skills"].append({
                "learner_id": f"L{i:04d}",
                "competency_id": item["competency_id"],
                "skill_name": item["competency_name"],
                "proficiency": rng.choice(
                    ["beginner", "developing", "proficient"]
                ),
            })

    job_titles = [
        "Junior Developer", "Backend Engineer", "Frontend Developer",
        "Data Analyst", "Machine Learning Engineer", "Cloud Engineer",
        "DevOps Engineer", "Full Stack Developer", "Software Engineer",
        "AI Engineer", "Database Developer", "Platform Engineer",
    ]
    for i in range(1, 151):
        primary = rng.choice(competencies)
        entities["jobs"].append({
            "job_id": f"J{i:04d}",
            "job_title": rng.choice(job_titles),
            "domain": primary["domain"],
            "description": (
                f"Work on {primary['competency_name']} and related "
                f"{primary['domain'].lower()} responsibilities."
            ),
            "primary_competency_id": primary["competency_id"],
        })
        selected = [primary] + rng.sample(
            [c for c in competencies if c != primary], 4
        )
        for item in selected:
            entities["job_requirements"].append({
                "job_id": f"J{i:04d}",
                "competency_id": item["competency_id"],
                "requirement": item["competency_name"],
                "importance": rng.choice(
                    ["essential", "important", "optional"]
                ),
            })

    course_topics = [
        "fundamentals", "practical projects", "advanced techniques",
        "hands-on workshop", "implementation guide", "core concepts",
    ]
    for i in range(1, 601):
        primary = rng.choice(competencies)
        entities["courses"].append({
            "course_id": f"CR{i:04d}",
            "course_title": (
                f"{primary['competency_name']}: "
                f"{rng.choice(course_topics)}"
            ),
            "domain": primary["domain"],
            "description": (
                f"A learning resource covering "
                f"{primary['competency_name']}."
            ),
            "level": rng.choice(["beginner", "intermediate", "advanced"]),
            "primary_competency_id": primary["competency_id"],
        })
        selected = [primary] + rng.sample(
            [c for c in competencies if c != primary], 1
        )
        for item in selected:
            entities["course_competencies"].append({
                "course_id": f"CR{i:04d}",
                "competency_id": item["competency_id"],
                "competency_name": item["competency_name"],
            })

    return entities


def skill_text(skill, rng, *, allow_bare=True):
    aliases = SKILL_ALIASES.get(skill, [])
    options = [
        f"experience with {skill}",
        f"knowledge of {skill}",
        f"ability to use {skill}",
    ] + list(aliases)
    if allow_bare:
        options.insert(0, skill)
    return rng.choice(options)


def make_pairs(rng, competencies, entities, target_count):
    skills_by_domain = defaultdict(list)
    for c in competencies:
        skills_by_domain[c["domain"]].append(c["competency_name"])

    skills = [c["competency_name"] for c in competencies]
    competency_by_name = {c["competency_name"]: c for c in competencies}

    learner_skills = defaultdict(list)
    for row in entities["learner_skills"]:
        learner_skills[row["learner_id"]].append(row["skill_name"])

    job_requirements = defaultdict(list)
    for row in entities["job_requirements"]:
        job_requirements[row["job_id"]].append(row["requirement"])

    course_competency_by_id = {
        row["course_id"]: row["competency_name"]
        for row in entities["course_competencies"]
    }

    pairs = []
    seen = set()

    target_labels = ["EXACT", "PARAPHRASE", "PARTIAL_MATCH", "RELATED", "PREREQUISITE", "UNRELATED"]
    label_weights = [0.15, 0.22, 0.22, 0.20, 0.06, 0.15]

    task_names = ["skill_competency", "learner_job", "gap_course", "relationship"]
    task_weights = [0.70, 0.18, 0.10, 0.02]

    attempts = 0
    max_attempts = target_count * 25
    while len(pairs) < target_count and attempts < max_attempts:
        attempts += 1
        desired_label = rng.choices(target_labels, weights=label_weights, k=1)[0]
        task = rng.choices(task_names, weights=task_weights, k=1)[0]

        if task in {"skill_competency", "relationship"}:
            if desired_label == "EXACT":
                a = rng.choice(skills)
                b = a
                text_a, text_b = a, a
            elif desired_label == "PARAPHRASE":
                a = rng.choice(skills)
                b = a
                text_a = skill_text(a, rng, allow_bare=True)
                text_b = skill_text(b, rng, allow_bare=False)
                # Ensure texts are strictly non-identical for paraphrases
                while text_a == text_b:
                    text_b = skill_text(b, rng, allow_bare=False)
            elif desired_label == "PREREQUISITE":
                domain = rng.choice(list(skills_by_domain.keys()))
                d_skills = skills_by_domain[domain]
                if len(d_skills) < 2:
                    continue
                a, b = rng.sample(d_skills, 2)
                text_a, text_b = skill_text(a, rng), skill_text(b, rng)
            elif desired_label in {"PARTIAL_MATCH", "RELATED"}:
                domain = rng.choice(list(skills_by_domain.keys()))
                d_skills = skills_by_domain[domain]
                if len(d_skills) < 2:
                    continue
                a, b = rng.sample(d_skills, 2)
                text_a, text_b = skill_text(a, rng), skill_text(b, rng)
            else:  # UNRELATED
                dom1, dom2 = rng.sample(list(skills_by_domain.keys()), 2)
                a = rng.choice(skills_by_domain[dom1])
                b = rng.choice(skills_by_domain[dom2])
                text_a, text_b = skill_text(a, rng), skill_text(b, rng)

            entity_a = competency_by_name[a]["competency_id"]
            entity_b = competency_by_name[b]["competency_id"]
            group = entity_a
            label = desired_label

        elif task == "learner_job":
            learner = rng.choice(entities["learners"])
            job = rng.choice(entities["jobs"])
            owned = set(learner_skills[learner["learner_id"]])
            required = set(job_requirements[job["job_id"]])
            overlap = owned & required

            text_a = f"Learner skills: {', '.join(sorted(owned)[:4])}"
            text_b = f"Job requirements: {', '.join(sorted(required)[:4])}"

            if overlap:
                label = "PARTIAL_MATCH" if desired_label in {"PARTIAL_MATCH", "UNRELATED"} else "PARAPHRASE"
            else:
                label = "UNRELATED"

            entity_a = learner["learner_id"]
            entity_b = job["job_id"]
            group = job["primary_competency_id"]

        else:  # gap_course
            learner = rng.choice(entities["learners"])
            course = rng.choice(entities["courses"])
            owned = set(learner_skills[learner["learner_id"]])
            course_skill = course_competency_by_id.get(
                course["course_id"],
                course["course_title"].split(":")[0],
            )

            if course_skill in owned:
                label = "RELATED"
            elif desired_label == "UNRELATED":
                label = "UNRELATED"
            else:
                label = "PARTIAL_MATCH"

            text_a = f"Skills known: {', '.join(sorted(owned)[:4])}"
            text_b = course["course_title"]
            entity_a = learner["learner_id"]
            entity_b = course["course_id"]
            group = course["course_id"]

        score = LABEL_SCORES[label]
        signature = (text_a, text_b)
        if signature in seen or text_a == "" or text_b == "":
            continue
        seen.add(signature)

        pairs.append({
            "pair_id": f"P{len(pairs) + 1:06d}",
            "task_type": task,
            "text_a": text_a,
            "text_b": text_b,
            "entity_a_id": entity_a,
            "entity_b_id": entity_b,
            "relationship_label": label,
            "relevance_score": score,
            "generation_group_id": str(group),
        })

    return pairs, 0


def validate(output_dir, entities, pairs, competencies,
             expected_learners, fallback_count):
    errors, warnings = [], []

    competency_ids = {c["competency_id"] for c in competencies}
    learner_ids = {r["learner_id"] for r in entities["learners"]}
    job_ids = {r["job_id"] for r in entities["jobs"]}
    course_ids = {r["course_id"] for r in entities["courses"]}
    valid_entity_ids = competency_ids | learner_ids | job_ids | course_ids

    if len(pairs) != len({p["pair_id"] for p in pairs}):
        errors.append("Duplicate pair_id values detected.")

    allowed = set(LABEL_SCORES)
    seen_text = set()
    for p in pairs:
        if p["relationship_label"] not in allowed:
            errors.append(f"Invalid label on {p['pair_id']}: {p['relationship_label']}")
        if p["relevance_score"] != LABEL_SCORES[p["relationship_label"]]:
            errors.append(f"Score/label mismatch on {p['pair_id']}")
        if p["relationship_label"] == "PARAPHRASE" and p["text_a"] == p["text_b"]:
            errors.append(f"Paraphrase with identical text on {p['pair_id']}")
        if p["entity_a_id"] not in valid_entity_ids:
            errors.append(f"Unknown entity_a_id on {p['pair_id']}")
        if p["entity_b_id"] not in valid_entity_ids:
            errors.append(f"Unknown entity_b_id on {p['pair_id']}")
        sig = (p["text_a"], p["text_b"])
        if sig in seen_text:
            warnings.append(f"Duplicate text pair at {p['pair_id']}")
        seen_text.add(sig)

    label_counts = Counter(p["relationship_label"] for p in pairs)
    task_counts = Counter(p["task_type"] for p in pairs)

    if pairs:
        top_label, top_n = label_counts.most_common(1)[0]
        if top_n / len(pairs) > 0.40:
            warnings.append(
                f"Label '{top_label}' dominates ({top_n/len(pairs):.0%})."
            )

    report = {
        "dataset_version": VERSION,
        "counts": {name: len(rows) for name, rows in entities.items()},
        "semantic_pairs": len(pairs),
        "relationship_counts": dict(label_counts),
        "task_counts": dict(task_counts),
        "fallback_template_pairs": fallback_count,
        "alias_coverage": alias_coverage(),
        "errors": errors,
        "warnings": warnings,
        "seed": SEED,
    }

    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if errors:
        raise RuntimeError(f"Dataset validation failed: {errors}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic semantic skill matching datasets."
    )
    parser.add_argument("--output", default="DataSet",
                        help="Directory for generated CSV files.")
    parser.add_argument("--pairs", type=int, default=25000,
                        help="Number of semantic pairs to generate.")
    parser.add_argument("--learners", type=int, default=500,
                        help="Number of synthetic learner profiles.")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    competencies = make_catalog()
    entities = build_entities(rng, competencies, learner_count=args.learners)
    pairs, fallback_count = make_pairs(rng, competencies, entities, args.pairs)

    write_csv(output_dir / "competencies.csv", competencies)
    for name, rows in entities.items():
        if rows:
            write_csv(output_dir / f"{name}.csv", rows)
        else:
            write_csv(output_dir / f"{name}.csv", [], fieldnames=[])

    write_csv(output_dir / "semantic_pairs.csv", pairs)
    validate(
        output_dir, entities, pairs, competencies,
        expected_learners=args.learners,
        fallback_count=fallback_count
    )

if __name__ == "__main__":
    main()