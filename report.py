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

def generate_pdf_report(image, predicted_class, confidence, predictions, class_names, heatmap_img=None, patient_name="", patient_age="", doctor_name=""):

    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    
    # We added "Margins" here. Margins are like a fence around the edge of the paper 
    # so the text doesn't touch the sides.
    doc = SimpleDocTemplate(temp_pdf.name, pagesize=letter, 
                            rightMargin=72, leftMargin=72, 
                            topMargin=72, bottomMargin=36)
    story = []

    # 1. Main Title
    title_style = ParagraphStyle("title", fontSize=26, fontName="Helvetica-Bold", alignment=1)
    story.append(Paragraph("NeuraSight Report", title_style))
    story.append(Spacer(1, 0.1*inch))

    # 2. Subtitle
    sub_style = ParagraphStyle("sub", fontSize=12, fontName="Helvetica", alignment=1, textColor=colors.HexColor("#667eea"))
    story.append(Paragraph("AI Brain Tumor Detection", sub_style))
    
    # BIGGER SPACER: 0.4 inches of invisible space to prevent crashing
    story.append(Spacer(1, 0.4*inch)) 

    # 3. Patient Details (Now organized beautifully inside a grid/table)
    date_str = datetime.now().strftime("%d/%m/%Y")
    
    # If the user didn't type a name, we put "Not Provided"
    p_name = patient_name if patient_name else "Not Provided"
    p_age = str(patient_age) + " years" if patient_age else "Not Provided"
    d_name = doctor_name if doctor_name else "Not Provided"

    patient_data = [
        ["Date:", date_str, "Doctor:", d_name],
        ["Patient Name:", p_name, "Patient Age:", p_age]
    ]

    patient_table = Table(patient_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), # Makes "Date" and "Patient Name" bold
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'), # Makes "Doctor" and "Patient Age" bold
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8f9fa")), # Very light gray background
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#d3d3d3")), # Draws a light border around the box
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(patient_table)
    
    # BIGGER SPACER
    story.append(Spacer(1, 0.4*inch))

    # 4. Result
    result_style = ParagraphStyle("result", fontSize=18, fontName="Helvetica-Bold", alignment=1)
    story.append(Paragraph("Result: " + predicted_class, result_style))
    story.append(Spacer(1, 0.1*inch))
    
    conf_style = ParagraphStyle("conf", fontSize=12, fontName="Helvetica", alignment=1)
    story.append(Paragraph("Confidence: " + str(round(confidence, 1)) + "%", conf_style))
    
    # BIGGER SPACER
    story.append(Spacer(1, 0.3*inch))

    # 5. MRI image (Made the image slightly bigger so it is clearer)
    mri_temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.resize((250, 250)).save(mri_temp.name)
    story.append(RLImage(mri_temp.name, width=3*inch, height=3*inch))
    
    # BIGGER SPACER
    story.append(Spacer(1, 0.4*inch))

    # 6. Probability table
    prob_data = [["Tumor Type", "Probability"]]
    for cls, prob in zip(class_names, predictions):
        prob_data.append([cls, str(round(prob * 100, 1)) + "%"])

    t = Table(prob_data, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f0ff"), colors.white]),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    
    # BIGGER SPACER
    story.append(Spacer(1, 0.4*inch))

    # 7. Clinical info
    class_info = {
        "Glioma": ("High", "Immediate neurological consultation required.", "Headaches, seizures, memory loss"),
        "Meningioma": ("Medium", "Neurosurgeon appointment within 1-2 weeks.", "Headaches, vision problems"),
        "No Tumor": ("None", "Continue regular health checkups.", "N/A"),
        "Pituitary": ("Low-Medium", "Endocrinology consultation recommended.", "Vision changes, hormonal issues")
    }

    info = class_info[predicted_class]
    clin_data = [
        ["Severity", info[0]],
        ["Recommended Action", info[1]],
        ["Common Symptoms", info[2]]
    ]
    clin_table = Table(clin_data, colWidths=[2*inch, 3.5*inch])
    clin_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0ff")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(clin_table)
    
    # BIGGER SPACER
    story.append(Spacer(1, 0.4*inch))

    # 8. Disclaimer
    disc = ParagraphStyle("disc", fontSize=8, fontName="Helvetica-Oblique", textColor=colors.grey, alignment=1)
    story.append(Paragraph("DISCLAIMER: For research only. Not medical advice. Always consult a qualified doctor.", disc))

    doc.build(story)
    return temp_pdf.name