import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

NUM_COURSES_TOTAL = 13
AVG_EXAMS_PER_SESSION = 3
NUM_SESSIONS_PER_YEAR = 5
ACCURACY = 0.98
MAX_NUM_STUDENTS = 10000
CONF_LEVEL = 0.98 #confidence level

#set random seed
np.random.seed(4)

#Exams distribution
DISTRIBUTION_6_CFU = [16, 10, 11, 16, 12, 13, 15, 12, 9, 8, 10, 3, 0]
DISTRIBUTION_8_CFU = [17, 4, 4, 5, 2, 9, 4, 6, 7, 14, 12, 19, 44]
DISTRIBUTION_10_CFU = [13, 0, 5, 7, 5, 8, 9, 12, 10, 22, 27, 28, 57]
PROB_6 = 135/241
PROB_8 = 147/225
PROB_10 = 203/301
SEMESTER_1 = [8, 8, 8, 8]
SEMESTER_2 = [6, 8, 10, 8]
SEMESTER_3 = [6, 8, 8, 6, 6]
TOTAL_CFU = 98

#extracting the number of exams taken is each session based on a poisson discrete distribution
def get_num_exams(num_courses_left):
  n = num_courses_left+1
  while n > num_courses_left:
    n = st.poisson.rvs(AVG_EXAMS_PER_SESSION)
  return n

#extracting the probability of passing each exam from a uniform distribution
def pass_not_pass():
  p = st.uniform.rvs()
  return p

#compute the grade distribution based on the given real distribution
def extract_grade(probabilities):
  grades = range(18, 31)
  tot = sum(probabilities)
  for i in range(len(probabilities)):
    probabilities[i] = probabilities[i]/tot
  g = np.random.choice(grades, p=probabilities)
  return g
#improve on this function to allow for different distributions for the different exams

#compute the actual graduation grade, including thesis, presentation, and bonus points
#we consider these additional point no longer independent, but with varrying distributions based on the average grade and graduation time
def compute_graduation_grade(avg_grade, graduation_time):
  #the points for the thesis depend on the average grade, if a student has a higher average, we can assume they will produce a better thesis
  #we can get between 0 and 4 points for the thesis
  if avg_grade <= 22:
    thesis = st.uniform.rvs(scale=2) #we get between 0 and 2 points
  elif (avg_grade > 22) & (avg_grade <= 26):
    thesis = st.uniform.rvs(loc=2, scale=1) #we get between 2 and 3 points
  elif (avg_grade > 26) & (avg_grade <= 30):
    thesis = st.uniform.rvs(loc=3, scale=1) #we get between 3 and 4 points
  #the points for the presentation we still consider independent from the other factors
  presentation = st.uniform.rvs(scale=2)
  #we decide to assign the bonus points based on the time of graduation, if a student graduates on time, they get more points
  #we can get between 0 and 2 bonus points
  if graduation_time == 2: #if 2 years we get between 1 and 2 points
    bonus = st.uniform.rvs(loc=1, scale=1)
  elif (graduation_time > 2) & (graduation_time <=4): #up to 4 years we get between 0 and 1 point
    bonus = st.uniform.rvs()
  elif graduation_time > 4: #if more no bonus points
    bonus = 0
  final_grade = (avg_grade/30)*110 + thesis + presentation + bonus
  if final_grade > 112.5:
    return 111 #we use 111 as a proxy for 110 cum laude
  else:
    return int(final_grade)

#we need to set a threshold for rejecting the proposed grade, it could be reasonable to assume that there is 
# a certain percentage of students that do not reject any grade (so the threshold is 18), and other students 
# that do have a certain threshold variable until 30. It is also reasonable to assume that only very few 
# students have very high thresholds, so the higher values of the threshold are extracted with lower probability
def extract_threshold():
  grade_and_base_increase = [
    (18, 0), (19, 0.01), (20, 0.02), (21, 0.03), (22, 0.04), (23, 0.05), (24, 0.06), 
    (25, 0.07), (26, 0.08), (27, 0.09), (28, 0.10), (29, 0.11), (30, 0.12)
  ]
  grades = range(18, 31)
  probs = [0.20, 0.02, 0.02, 0.03, 0.04, 0.06, 0.10, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
  th = np.random.choice(grades, p=probs)
  for i in range(len(grade_and_base_increase)):
    if grade_and_base_increase[i][0] == th:
      threshold = grade_and_base_increase[i][0]
      increase = grade_and_base_increase[i][1]
  return threshold, increase

# we extract the level of preparedness for the specific exam: it is a number from 1 to 5, which potentially 
# increases the probability of passing the exam
def extract_preparedness():
  prep = np.random.choice(range(1, 6))
  if prep == 1:
    return prep, 0
  elif prep == 2:
    return prep, 0.01
  elif prep == 3:
    return prep, 0.025
  elif prep == 4:
    return prep, 0.05
  elif prep == 5:
    return prep, 0.1

# simulate the career for a single student
# we assume at the start of the first session is the first semester, at the start of the third is 
# the second semester, at the start of the sixth is the third semester
def simulate_student_career():
  #define for the student a certain rejection threshold
  rejection_th, base_increase = extract_threshold()
  initial_th = rejection_th
  num_courses_left = NUM_COURSES_TOTAL
  grades = []
  weighted_grades = []
  exams = []
  num_sessions = 0
  session_counter = 0
  #print('Rejection threshold:', rejection_th)
  #simulate each session (corresponding to one iteration of the loop)
  while num_courses_left > 0:
    num_sessions += 1 #start of a new session
    #print('Session', num_sessions)
    if num_sessions == 1: #we only consider in the exams we can take in the current session, those of the semester that just ended or the previous ones
      exams += SEMESTER_1
    elif num_sessions == 3:
      exams += SEMESTER_2
    elif num_sessions == 6:
      exams += SEMESTER_3
    #print('Exams we can take:', exams)
    #we decrease the rejection threshold every three years
    if num_sessions == session_counter + 3:
      session_counter = num_sessions
      if rejection_th > 18:
        rejection_th -= 1
        #print('Reducing the threshold to', rejection_th)
    #extract from distribution with specified average the number of exams we will take in a session
    num_exams = get_num_exams(len(exams))
    for _ in range(num_exams):
      #extract which exam we will try (by extracting a random index from the exams list)
      exam_index = np.random.choice(range(len(exams)))
      current_exam = exams[exam_index]
      #print('Trying exam', current_exam)
      #extract the preparedness of the student for the exam, which will determine an increase in the probability of passing
      prep, increase = extract_preparedness()
      #print('Preparedness level:', prep)
      #extract if the exam is passed or not from the given probability
      p = pass_not_pass()
      p_increased = p + base_increase + increase
      if p_increased > 1:
        p_increased = 1
      #print('Probability of passing:', p_increased)
      #check if the exam is passed or not by checking the right probability and distribution for the number of cfu
      #  of the extracted exam
      if current_exam == 6:
        if p > PROB_6:
          proposed_grade = extract_grade(DISTRIBUTION_6_CFU) #extract the grade from the given distribution
          #print('Proposed grade:', proposed_grade)
          #if the proposed grade is above the threshold, we register the grade, remove the exam from the list, and decrease the n of exams left
          if proposed_grade >= rejection_th:
            #print('Accepting the grade')
            grades.append(proposed_grade)
            weighted_grades.append(proposed_grade * current_exam)
            exams.pop(exam_index)
            num_courses_left -= 1
          else:
            #print('Grade rejected')
            pass
        else:
          #print('Exam failed')
          pass
      elif current_exam == 8:
        if p > PROB_8:
          proposed_grade = extract_grade(DISTRIBUTION_8_CFU) #extract the grade from the given distribution
          #print('Proposed grade:', proposed_grade)
          #if the proposed grade is above the threshold, we register the grade,
          #  remove the exam from the list, and decrease the n of exams left
          if proposed_grade >= rejection_th:
            #print('Accepting the grade')
            grades.append(proposed_grade)
            weighted_grades.append(proposed_grade * current_exam)
            exams.pop(exam_index)
            num_courses_left -= 1
          else:
            #print('Grade rejected')
            pass
        else:
          #print('Exam failed')
          pass
      elif current_exam == 10:
        if p > PROB_10:
          proposed_grade = extract_grade(DISTRIBUTION_10_CFU) #extract the grade from the given distribution
          #print('Proposed grade:', proposed_grade)
          #if the proposed grade is above the threshold, we register the grade, remove the exam from the list, and decrease the n of exams left
          if proposed_grade >= rejection_th:
            #print('Accepting the grade')
            grades.append(proposed_grade)
            weighted_grades.append(proposed_grade * current_exam)
            exams.pop(exam_index)
            num_courses_left -= 1
          else:
            #print('Grade rejected')
            pass
        else:
          #print('Exam failed')
          pass

  #compute the average of the grades
  avg_grade = round(sum(grades)/NUM_COURSES_TOTAL)
  #compute the weighted average of the grades
  weighted_avg = round(sum(weighted_grades)/TOTAL_CFU)
  #compute the graduation time in years
  if num_sessions%NUM_SESSIONS_PER_YEAR == 0:
    graduation_time = int(num_sessions/NUM_SESSIONS_PER_YEAR)
  else:
    graduation_time = int(num_sessions/NUM_SESSIONS_PER_YEAR +1)
  #print('Average of the grades', avg_grade)
  #print('Graduation time:', graduation_time)
  #compute the actual graduation grade, considering thesis, presentation and bonus
  actual_graduation_grade = compute_graduation_grade(weighted_avg, graduation_time)
  #print('Graduation grade:', actual_graduation_grade)
  return actual_graduation_grade, graduation_time, initial_th

#function that computes the confidence interval and the accuracy for the given elements in x
#n is the number of elements, c is the confidence level
def compute_conf_int(x, n, c):
  #computing the empirical average
  avg = np.mean(x)
  #computing the confidence interval
  if n<30: #for n of samples less then 30 we use the t-student distribution
    I = st.t.interval(c, n-1, avg, st.sem(x))
  elif n >= 30: #for more than 30 samples we use the normal distribution
    I = st.norm.interval(c, avg, st.sem(x))
  #computing the interval width
  w = (I[1]-I[0])/2
  #computing the relative error
  e = w/avg
  #computing the accuracy
  acc = 1-e
  return (I, acc)

np.random.seed(4)
#simulate for various failure probabilities
#print('Probability of failure:', fail_prob)
#repeat the simulation for a single student a number of times to compute averages and the relative confidence intervals
grad_grades = []
grad_times = []
rej_thresholds = []
acc_grad_grade = 0
acc_grad_time = 0

#repeat the simulation until all the accuracies are above the one specified in the parameters
#for n in range(MAX_NUM_STUDENTS):
n=0
while min(acc_grad_grade, acc_grad_time) < ACCURACY:
  grad_grade, grad_time, rejection_th = simulate_student_career()
  grad_grades.append(grad_grade)
  grad_times.append(grad_time)
  rej_thresholds.append(rejection_th)
  if n > 1:
    _, acc_grad_grade = compute_conf_int(grad_grades, n, CONF_LEVEL)
    _, acc_grad_time = compute_conf_int(grad_times, n, CONF_LEVEL)
    #print(acc_avg_grade, acc_grad_grade, acc_grad_time)
  if (acc_grad_grade > ACCURACY) & (acc_grad_time > ACCURACY):
    print('Simulated for', n, 'students')
    print('Average graduation grade:', np.mean(grad_grades))
    print('Average graduation time:', np.mean(grad_times))
    break
  n+=1

#computing the average graduation grade for each time duration
grades_per_years = np.zeros(max(grad_times))
for i in range(len(grad_times)):
  grades_per_years[grad_times[i]-1] += grad_grades[i]
for i in range(max(grad_times)):
  grades_per_years[i] = grades_per_years[i]/grad_times.count(i+1)
#print(grades_per_years)

#computing  the average graduation grade for each initial rejection threshold
grades_per_th = np.zeros(max(rej_thresholds))
for i in range(len(rej_thresholds)):
  grades_per_th[rej_thresholds[i]-1] += grad_grades[i]
for i in range(max(rej_thresholds)):
  if rej_thresholds.count(i+1) > 0:
    grades_per_th[i] = grades_per_th[i]/rej_thresholds.count(i+1)
#print(grades_per_th)

#computing  the average graduation time for each initial rejection threshold
times_per_th = np.zeros(max(rej_thresholds))
for i in range(len(rej_thresholds)):
  times_per_th[rej_thresholds[i]-1] += grad_times[i]
for i in range(max(rej_thresholds)):
  if rej_thresholds.count(i+1) > 0:
    times_per_th[i] = times_per_th[i]/rej_thresholds.count(i+1)
#print(times_per_th)

#plot distribution of graduation grades (histogram)
#plot grad time distribution (histogram)
#plot grade vs time
plt.figure()
plt.hist(grad_grades, bins=(max(grad_grades)-min(grad_grades)))
plt.xlabel('Graduation grades distribution')
plt.xticks(range(min(grad_grades), max(grad_grades)+1), fontsize=7)
plt.yticks(fontsize=7)
plt.show()
plt.figure()
plt.hist(grad_times, bins=(max(grad_times)-min(grad_times)))
plt.xlabel('Graduation time distribution (in years)')
plt.xticks(range(min(grad_times), max(grad_times)+1))
plt.show()
plt.figure()
plt.plot(range(1, max(grad_times)+1), grades_per_years, marker='o')
plt.xlabel('Graduation time (in years)')
plt.ylabel('Average graduation grade')
plt.grid()
plt.show()
plt.figure()
plt.plot(range(1, max(rej_thresholds)+1), grades_per_th, marker='o')
plt.xlim([18, 30])
plt.xlabel('Initial rejection threshold')
plt.ylabel('Average graduation grade')
plt.grid()
plt.show()
plt.figure()
plt.plot(range(1, max(rej_thresholds)+1), times_per_th, marker='o')
plt.xlim([18, 30])
plt.xlabel('Initial rejection threshold')
plt.ylabel('Average time to graduate (in years)')
plt.grid()
plt.show()