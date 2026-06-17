from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image
import tempfile
import os
from datetime import datetime


def generate_pdf_report(image, predicted_class, confidence, predictions, class_names, heatmap_img=None):

    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc = SimpleDocTemplate(temp_pdf.name, pagesize=letter)
    story = []

    title_style = ParagraphStyle("title", fontSize=24, fontName="Helvetica-Bold", alignment=1)
    story.append(Paragraph("NeuraSight Report", title_style))
    story.append(Spacer(1, 0.2*inch))

    sub_style = ParagraphStyle("sub", fontSize=11, fontName="Helvetica", alignment=1)
    story.append(Paragraph("AI Brain Tumor Detection", sub_style))
    story.append(Spacer(1, 0.2*inch))

    date_style = ParagraphStyle("date", fontSize=10, fontName="Helvetica")
    story.append(Paragraph("Date: " + datetime.now().strftime("%d/%m/%Y"), date_style))
    story.append(Spacer(1, 0.1*inch))

    result_style = ParagraphStyle("result", fontSize=16, fontName="Helvetica-Bold", alignment=1)
    story.append(Paragraph("Result: " + predicted_class, result_style))
    story.append(Paragraph("Confidence: " + str(round(confidence, 1)) + "%", sub_style))
    story.append(Spacer(1, 0.2*inch))

    mri_temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.resize((200, 200)).save(mri_temp.name)
    story.append(RLImage(mri_temp.name, width=2.5*inch, height=2.5*inch))
    story.append(Spacer(1, 0.2*inch))

    prob_data = [["Type", "Probability"]]
    for cls, prob in zip(class_names, predictions):
        prob_data.append([cls, str(round(prob * 100, 1)) + "%"])

    t = Table(prob_data)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.blue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    disc = ParagraphStyle("disc", fontSize=8, fontName="Helvetica-Oblique", textColor=colors.grey)
    story.append(Paragraph("DISCLAIMER: For research only. Not medical advice. Consult a doctor.", disc))

    doc.build(story)
    return temp_pdf.name