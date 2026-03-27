import os
from fpdf import FPDF

class PDF(FPDF):
    def __init__(self):
        super().__init__(unit="mm", format="A4")
        self.set_margins(20, 20, 20)
        self.set_auto_page_break(auto=True, margin=20)
        
        # Load fonts
        arial_path = "C:/Windows/Fonts/arial.ttf"
        arial_bold_path = "C:/Windows/Fonts/arialbd.ttf"
        self.add_font("Arial", "", arial_path)
        self.add_font("Arial", "B", arial_bold_path)

    def header(self):
        if self.page_no() == 1:
            self.set_font("Arial", "B", 16)
            self.cell(0, 10, "BÁO CÁO DỰ ÁN BUSINESS INTELLIGENCE", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "", 8)
        self.cell(0, 10, f"Trang {self.page_no()}", align="C")

def generate(input_file, output_file):
    pdf = PDF()
    pdf.add_page()
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
        
        # Header 1
        if line.startswith("# "):
            pdf.set_font("Arial", "B", 18)
            pdf.multi_cell(0, 10, line[2:])
            pdf.ln(2)
        # Header 2
        elif line.startswith("## "):
            pdf.set_font("Arial", "B", 14)
            pdf.multi_cell(0, 8, line[3:])
            pdf.ln(2)
        # Header 3
        elif line.startswith("### "):
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(0, 7, line[4:])
            pdf.ln(1)
        # List items
        elif line.startswith("- ") or line.startswith("* "):
            pdf.set_font("Arial", "", 11)
            pdf.write(6, "  • ")
            pdf.multi_cell(0, 6, line[2:].replace("**", ""))
        # Normal text
        else:
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 6, line.replace("**", ""))

    pdf.output(output_file)
    print(f"Success: {output_file}")

if __name__ == "__main__":
    generate("Final_Project_Report.md", "Business_Sentiment_Analysis_Report.pdf")
