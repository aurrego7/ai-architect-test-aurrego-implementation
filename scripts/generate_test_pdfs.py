"""Generate sample scanned-style PDF documents for testing.

Creates image-based PDFs (no text layer) so OCR is required to extract text.
Uses only PIL (Pillow) — no extra dependencies needed.
"""

import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_pdfs")

# Page dimensions at 150 DPI (letter size: 8.5 x 11 inches)
PAGE_WIDTH = 1275
PAGE_HEIGHT = 1650
MARGIN = 100
LINE_HEIGHT = 35
DPI = 150


def get_font(size: int = 20):
    """Get a font, falling back to default if system fonts unavailable."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_text_block(draw, lines, start_x, start_y, font, color="black"):
    """Draw multiple lines of text, return final y position."""
    y = start_y
    for line in lines:
        draw.text((start_x, y), line, fill=color, font=font)
        y += LINE_HEIGHT
    return y


def create_page_image(lines, title=None):
    """Create a single page image with text content."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    font = get_font(20)
    title_font = get_font(28)

    y = MARGIN

    if title:
        draw.text((MARGIN, y), title, fill="black", font=title_font)
        y += 60

    draw_text_block(draw, lines, MARGIN, y, font)
    return img


def generate_company_memo():
    """Generate a 1-page company memo with several names."""
    lines = [
        "MEMORANDUM",
        "",
        "To: All Department Heads",
        "From: Margaret Thompson, CEO",
        "Date: March 15, 2024",
        "Subject: Q1 Performance Review and Team Restructuring",
        "",
        "Dear Team,",
        "",
        "I am writing to inform you of several organizational changes",
        "effective immediately. After careful review by our executive",
        "committee, the following appointments have been made:",
        "",
        "1. Robert Chen has been promoted to Vice President of Engineering.",
        "   He will oversee all technical operations moving forward.",
        "",
        "2. Dr. Sarah Williams will join as Chief Data Scientist,",
        "   reporting directly to the CTO, James Anderson.",
        "",
        "3. Maria Garcia has been appointed as Head of Product Design.",
        "   She previously led the UX team at our London office.",
        "",
        "4. The new project codenamed 'Atlas' will be managed by",
        "   David Nakamura and his team of senior architects.",
        "",
        "Please direct any questions to Patricia Okonkwo in Human",
        "Resources or contact me directly.",
        "",
        "Best regards,",
        "Margaret Thompson",
        "Chief Executive Officer",
    ]

    img = create_page_image(lines)
    return [img]


def generate_meeting_minutes():
    """Generate a 2-page meeting minutes document."""
    page1_lines = [
        "BOARD OF DIRECTORS - MEETING MINUTES",
        "",
        "Date: January 22, 2024",
        "Location: Conference Room A, Headquarters",
        "Time: 10:00 AM - 12:30 PM",
        "",
        "Attendees:",
        "  - Richard Hernandez, Chairman of the Board",
        "  - Elizabeth Park, Chief Financial Officer",
        "  - Thomas Muller, Chief Technology Officer",
        "  - Dr. Aisha Patel, Independent Director",
        "  - Kevin O'Brien, General Counsel",
        "  - Jennifer Liu, Secretary",
        "",
        "Absent:",
        "  - Carlos Mendoza, VP of Sales (excused)",
        "",
        "1. CALL TO ORDER",
        "",
        "Richard Hernandez called the meeting to order at 10:05 AM.",
        "Jennifer Liu confirmed quorum was present.",
        "",
        "2. APPROVAL OF PREVIOUS MINUTES",
        "",
        "Elizabeth Park moved to approve the minutes from the",
        "December 2023 meeting. Thomas Muller seconded the motion.",
        "Motion carried unanimously.",
        "",
        "3. FINANCIAL REPORT",
        "",
        "Elizabeth Park presented the Q4 financial results.",
        "Revenue increased 12% year-over-year. Dr. Aisha Patel",
        "raised concerns about rising operational costs in the",
        "APAC region. Kevin O'Brien noted potential regulatory",
        "implications of the proposed expansion into Southeast Asia.",
    ]

    page2_lines = [
        "4. TECHNOLOGY UPDATE",
        "",
        "Thomas Muller provided an update on the AI initiative.",
        "He reported that the machine learning team, led by",
        "Dr. Yuki Tanaka, has achieved significant milestones",
        "in the natural language processing project.",
        "",
        "Alexander Popov from the infrastructure team presented",
        "the cloud migration timeline. The full migration to AWS",
        "is expected to be completed by Q3 2024.",
        "",
        "5. NEW BUSINESS",
        "",
        "Richard Hernandez introduced a proposal for a strategic",
        "partnership with NovaTech Solutions. The partnership would",
        "be managed by Catherine Dubois from Business Development.",
        "",
        "Dr. Aisha Patel recommended conducting due diligence",
        "before proceeding. Kevin O'Brien agreed to coordinate",
        "the legal review with external counsel.",
        "",
        "6. ADJOURNMENT",
        "",
        "Richard Hernandez adjourned the meeting at 12:25 PM.",
        "The next meeting is scheduled for March 18, 2024.",
        "",
        "Respectfully submitted,",
        "Jennifer Liu",
        "Corporate Secretary",
        "",
        "Approved by: Richard Hernandez, Chairman",
    ]

    img1 = create_page_image(page1_lines)
    img2 = create_page_image(page2_lines)
    return [img1, img2]


def generate_research_report():
    """Generate a 2-page research team report."""
    page1_lines = [
        "RESEARCH DEPARTMENT - ANNUAL REPORT 2024",
        "",
        "Prepared by: Dr. Olivia Chambers, Research Director",
        "",
        "EXECUTIVE SUMMARY",
        "",
        "This report summarizes the research activities and",
        "achievements of the department for fiscal year 2024.",
        "Our team of 45 researchers across three divisions has",
        "produced 23 peer-reviewed publications.",
        "",
        "TEAM LEADS AND KEY PERSONNEL",
        "",
        "Computational Biology Division:",
        "  Lead: Dr. Benjamin Foster",
        "  Senior Researchers: Dr. Priya Sharma, Lucas Zimmermann",
        "  Notable: Published groundbreaking work on protein folding",
        "",
        "Machine Learning Division:",
        "  Lead: Dr. Fatima Al-Rashidi",
        "  Senior Researchers: Christopher Wong, Anna Kowalski",
        "  Notable: Developed novel transformer architecture",
        "",
        "Applied Mathematics Division:",
        "  Lead: Prof. Michael O'Sullivan",
        "  Senior Researchers: Dr. Elena Volkov, Raj Krishnamurthy",
        "  Notable: Breakthrough in optimization algorithms",
        "",
        "EXTERNAL COLLABORATIONS",
        "",
        "Our department maintained active collaborations with",
        "several institutions. Dr. Priya Sharma co-authored a",
        "paper with Prof. Hans Weber from ETH Zurich.",
    ]

    page2_lines = [
        "Dr. Fatima Al-Rashidi presented at NeurIPS alongside",
        "Dr. James Chen from Stanford University.",
        "",
        "AWARDS AND RECOGNITION",
        "",
        "- Dr. Benjamin Foster received the Young Investigator",
        "  Award from the National Science Foundation.",
        "- Anna Kowalski was named to the Forbes 30 Under 30",
        "  list in the Science category.",
        "- Christopher Wong received a best paper award at ICML.",
        "",
        "BUDGET AND RESOURCES",
        "",
        "Total research budget: $4.2M",
        "External grants secured: $1.8M",
        "  - NIH Grant (PI: Dr. Benjamin Foster): $600K",
        "  - NSF Grant (PI: Dr. Fatima Al-Rashidi): $450K",
        "  - DARPA Contract (PI: Prof. Michael O'Sullivan): $750K",
        "",
        "LOOKING AHEAD",
        "",
        "For 2025, we plan to expand the ML division under",
        "Dr. Fatima Al-Rashidi. New hires will include two",
        "postdoctoral researchers and a visiting scholar.",
        "",
        "Submitted by:",
        "Dr. Olivia Chambers",
        "Director of Research",
        "",
        "Reviewed by: Margaret Thompson, CEO",
        "Date: December 15, 2024",
    ]

    img1 = create_page_image(page1_lines)
    img2 = create_page_image(page2_lines)
    return [img1, img2]


def save_as_pdf(images, filename):
    """Save list of PIL images as a single PDF."""
    path = os.path.join(OUTPUT_DIR, filename)
    if len(images) == 1:
        images[0].save(path, "PDF", resolution=DPI)
    else:
        images[0].save(
            path,
            "PDF",
            resolution=DPI,
            save_all=True,
            append_images=images[1:],
        )
    print(f"Created: {path} ({len(images)} page(s))")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Document 1: Single page memo (7 names)
    memo_pages = generate_company_memo()
    save_as_pdf(memo_pages, "company_memo.pdf")

    # Document 2: Two page meeting minutes (12 names)
    minutes_pages = generate_meeting_minutes()
    save_as_pdf(minutes_pages, "meeting_minutes.pdf")

    # Document 3: Two page research report (15+ names)
    report_pages = generate_research_report()
    save_as_pdf(report_pages, "research_report.pdf")

    # Print expected names for each document
    print("\n--- Expected Names ---")
    print("\ncompany_memo.pdf:")
    for name in [
        "Margaret Thompson",
        "Robert Chen",
        "Sarah Williams",
        "James Anderson",
        "Maria Garcia",
        "David Nakamura",
        "Patricia Okonkwo",
    ]:
        print(f"  - {name}")

    print("\nmeeting_minutes.pdf:")
    for name in [
        "Richard Hernandez",
        "Elizabeth Park",
        "Thomas Muller",
        "Aisha Patel",
        "Kevin O'Brien",
        "Jennifer Liu",
        "Carlos Mendoza",
        "Yuki Tanaka",
        "Alexander Popov",
        "Catherine Dubois",
    ]:
        print(f"  - {name}")

    print("\nresearch_report.pdf:")
    for name in [
        "Olivia Chambers",
        "Benjamin Foster",
        "Priya Sharma",
        "Lucas Zimmermann",
        "Fatima Al-Rashidi",
        "Christopher Wong",
        "Anna Kowalski",
        "Michael O'Sullivan",
        "Elena Volkov",
        "Raj Krishnamurthy",
        "Hans Weber",
        "James Chen",
        "Margaret Thompson",
    ]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
