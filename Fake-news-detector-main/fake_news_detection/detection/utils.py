import matplotlib.pyplot as plt
import io
import base64

def generate_dummy_pie_chart(label):
    labels = ['Fake', 'Real']
    sizes = [70, 30] if label == "Fake" else [30, 70]

    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f"Prediction: {label}")

    # Save the pie chart to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the image as base64 and return it
    return base64.b64encode(buf.getvalue()).decode()
