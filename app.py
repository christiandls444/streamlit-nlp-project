# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 


# Utils
import joblib 
pipe_lr = joblib.load(open("model/emotion_model.pkl","rb"))


# Track Utils


# Fxn
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


# Main Application
def main():
	
	menu = ["Home","About"]
	choice = st.sidebar.selectbox("",menu)

	if choice == "Home":
		
		st.title("Emotion Classifier")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2  = st.columns(2)

			# Apply Fxn Here
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)
			
			

			with col1:
				st.info("Original Text")
				st.write(raw_text)

				st.info("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}: {}".format(prediction,emoji_icon))
				st.write("Confidence: {}".format(np.max(probability)))



			with col2:
				st.info("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)



	# elif choice == "Monitor":
		
	# 	st.subheader("Monitor App")



	else:
		st.title("About")
		from PIL import Image
		img = Image.open("profile.png")
		st.image(img, width=200, caption="Source: DLSX")			
		st.markdown("##### Hi! I am Christian M. De Los Santos, Trained in Intermediate Data Analytics and Artificial Intelligence 101, "
					"Assuming Pythonista, AI/ML Enthusiast, I have an extensive background working with varied datasets especially Twitter. ")
		st.markdown("##### I am excited to keep learning new methods in a challenging and fast-paces environment.")



if __name__ == '__main__':
	main()