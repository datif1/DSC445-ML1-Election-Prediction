# Meeting Minutes for Project Group 5
Predicting U.S. County-Level Presidential Voting Outcomes Using Demographic and Socioeconomic Machine Learning Models

## Members
- Veon Ivon Almeida
- Shang Andrews
- Mohammed Atif Diwan  
- Yung Han Jeong
- Siddhesh Dayanand Patil

## Purpose
This document serves as the summary of meeting minutes, online discussion points, and action items for all members. In addition, it serves as a reference document for the peer evaluation to be performed at the end of the project.

## Meeting Minutes
### Meeting #1 – Introduction and Proposal Discussion (4/29/26)
Attendance: all
- Started MS Teams Group for discussion
- All project members met online on MS Teams for introduction
- Started a weekly meeting series Wednesday 3pm-4pm (CST) for weekly check-ins
- Determined few schedule challenges. Agreed to record all meetings for review for any members that cannot attend future meetings.
    - If anyone missed a meeting, they would review the meeting notes and participate in any discussion via MS Teams Group
- Initial project direction agreed on political dataset analysis with network analysis for gerrymandering analysis
    - Suggested focusing on Virginia and use their most recent redistricted voting districts for analysis
#### Action Items
    - All: Review and research additional dataset and share details for proposal writing on MS Teams Group

 
### Online Project Proposal Discussion (4/30/26 – 5/4/26) 
Attendance: all
- Project goal was pivoted to predicting presidential election (Democrat vs Republican) based on county level dataset. This pivot was agreed on after determining difficulty of creating robust dataset for gerrymandering analysis based on real-life dataset.
- Shang Andrews: Created first draft proposal based on the new project direction. Provided various edits and suggestions for the final draft.
- Mohammed Atif Diwan: Suggested additional data from MIT and US Census data. Provided additional details on EDA and modeling details for future tasks. Reviewed final draft and provide various suggestion and edits
- Siddhesh Dayanand Patil: Reviewed project proposals and dataset, provided additional details for future tasks
- Veon Ivon Almeida: Reviewed and provided various edits on the project proposal
- Yung Han Jeong: Organized and updated the project proposal for the final draft. Updated various sections with additional details and clarification and implemented various suggested edits from the team members.  

### Meeting #2 – Proposal Confirmation and EDA Next Steps (5/6/26)
Attendance: Shang Andrews, Mohammed Atif Diwan, Veon Ivon Almeida, Yung Han Jeong
- Project meeting time changed to 3:30-4:30 PM for future meetings
- Group review of the project proposal
- Determined next steps for individual EDA on dataset for modeling discussion and project goal division for future steps

#### Action items
- Yung Han Jeong: creation/update of this document to be sharable (moved to word doc on cloud)
- Shang Andrews/Mohammed Atif Diwan: Start github repository for project storage
- All: perform EDA individually on proposed dataset. Prepare discussion points for modeling next week. 

### Meeting #3 - EDA Findings and Model Discussions (5/14/26)
Attendance: Mohammed Atif Diwan, Veon Ivon Almeida, Yung Han Jeong
- Mohammed Atif Diwan completed github repository setup and invited all members
- Mohammed Atif Diwan completed EDA and published EDA report to github
- Discussed various data engineering and modeling steps
    - After EDA is completed all member will collaborate to finalize the training data including feature engineering
    - For completing the final report the all models should be predicted on 2024 election dataset
    - All members should build and train a model, if members choose the same type of model the final model selection should be done per model performance. For the final report one classification and one regression model should be chosen for the prediction. Other models can be included as the part of the analysis. 
    - Completed/engineered dataset should be split and each individual should take a slice of last 5 election cycles for training/additional training
        - Election results of 2008, 2012, 2016, 2020, and 2024 is set aside as prediction targets.
        - The model creator trains up to using 2004 dataset and predicts on 2008 and saves the model and model weights
        - The next member trains the saved model using 2008 dataset and predicts on 2012. 
        - This repeats until the final member achieves the 2024 results for the final results. 
- Action items
    - All: complete EDA and post findings as a markdown in report form to the github repository
    - All: from the EDA reports suggest data engineering methods
    - Mohammed Atif Diwan: begin data engineering on completed report and start classification modeling
    - Yung Han Jeong: complete EDA and data engineering task, start on regression modeling