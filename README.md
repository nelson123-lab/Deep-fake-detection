# Deepfake Video Detection

## Background: 

Deepfakes – audio, video, image, and text – are become a greater threat, and our ability to detect that something is a fake is limited.  In particular, with the greater reliance on video conferencing tools such as Zoom and WebEx, businesses are concerned that someone in appropriate can join a meeting without being noticed by using a deepfake of an expected employee.

## Primary goal for project: 

Algorithms to detect deepfakes of faces in videos exist, however they tend to assume a single face mostly centered in the screen.  For example, some research has focused on person-of-interest detection, where the characteristics of a particular individual (e.g., a politician) are learned, and then deepfakes are detected as deviations from these physical identifiers (where the identifiers are based on muscle movements that are unique to the individual).  Other approaches look for faces in a video and examine the area around the face for artifacts but can be confused with smaller faces in the background, generating false positives.  Neither approach converts well to video conferencing with multiple participants.

This project starts with a review of the relevant literature, determining if there is an algorithm that can potentially be adapted for this use case.  If no algorithm is found, the team will need to develop one.

Several datasets exist for training and testing.  The team will need to adapt these datasets to a zoom-like setting for both training and testing, assuming varying numbers of participants.  The team will need to demonstrate that their detection algorithm works when there are no deepfakes, as well as when there are one or multiple deepfakes.  A contribution to the research community would be to release this dataset publicly for other researchers to use.  

The team should also demonstrate that their algorithm generalizes to unseen circumstances.  One approach to doing this would be to train with, e.g., 2 types (datasets) of deepfakes, and then test against the other datasets. 

## Stretch goals:

1.	Can the detection algorithm be designed to run in real-time?  If so, a demo showing someone joining as a deepfake and an alert being shown on the screen would be exceptional.
2.	Demonstrating that the approach works on not only zoom, but other video conferencing, such as Webex, would also be desirable.
3.	Testing against different types of fakes – both face-swap and avatars – would also be desirable.
