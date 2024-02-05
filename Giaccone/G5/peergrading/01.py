import random
import statistics
from collections import Counter
import numpy as np
import math

################################## input parameters ##################################
################## probabilities of getting a certain grade for each course ##########

# mandatory courses and free ECTS courses
grade_probabilities_commmon_courses = {
    'data management and visualization': {
        18: 21, 19: 10, 20: 5, 21: 3, 22: 3, 23: 8, 24: 6, 25: 13, 26: 11, 27: 19, 28: 15, 29: 16, 30: 6
    },
    'data science lab: process and methods': {
        18: 4, 19: 9, 20: 5, 21: 7, 22: 9, 23: 11, 24: 12, 25: 8, 26: 19, 27: 10, 28: 8, 29: 11, 30: 16
    },
    'data ethics and data protection': {
        18: 9, 19: 7, 20: 4, 21: 11, 22: 8, 23: 6, 24: 11, 25: 11, 26: 7, 27: 6, 28: 9, 29: 1, 30: 0
    },
    'distributed architecture for big data processing and analytics': {
        18: 13, 19: 1, 20: 3, 21: 5, 22: 2, 23: 7, 24: 3, 25: 5, 26: 4, 27: 10, 28: 8, 29: 16, 30: 38
    },
    'machine learning and deep learning': {
        18: 11, 19: 0, 20: 4, 21: 4, 22: 5, 23: 4, 24: 6, 25: 7, 26: 6, 27: 18, 28: 22, 29: 23, 30: 47
    },
    'mathematics in machine learning': {
        18: 0, 19: 8, 20: 2, 21: 5, 22: 2, 23: 3, 24: 6, 25: 4, 26: 5, 27: 5, 28: 3, 29: 7, 30: 11
    },
    'innovation management': {
        18: 7, 19: 2, 20: 8, 21: 8, 22: 4, 23: 8, 24: 4, 25: 6, 26: 5, 27: 8, 28: 6, 29: 3, 30: 7
    },
    'bioquants': {
        18: 1, 19: 0, 20: 0, 21: 2, 22: 0, 23: 0, 24: 1, 25: 0, 26: 2, 27: 2, 28: 5, 29: 5, 30: 2
    },
    'cloud computing': {
        18: 3, 19: 2, 20: 4, 21: 4, 22: 8, 23: 10, 24: 9, 25: 10, 26: 6, 27: 3, 28: 5, 29: 5, 30: 9
    },
    'computational intelligence': {
        18: 0, 19: 0, 20: 1, 21: 1, 22: 2, 23: 1, 24: 1, 25: 2, 26: 2, 27: 4, 28: 3, 29: 0, 30: 59
    },
    'financial engineering': {
        18: 0, 19: 1, 20: 1, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 1, 27: 2, 28: 0, 29: 0, 30: 0
    },
    'human computer interaction': {
        18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 2, 27: 6, 28: 26, 29: 25, 30: 88
    },
    'information system': {
        18: 1, 19: 1, 20: 3, 21: 4, 22: 5, 23: 9, 24: 7, 25: 7, 26: 8, 27: 10, 28: 8, 29: 6, 30: 15
    },
    'information system security': {
        18: 24, 19: 8, 20: 11, 21: 11, 22: 7, 23: 15, 24: 13, 25: 7, 26: 6, 27: 8, 28: 11, 29: 2, 30: 4
    },
    'metodi quantitativi per la gestione del rischio': {
        18: 0, 19: 0, 20: 1, 21: 0, 22: 2, 23: 3, 24: 2, 25: 5, 26: 3, 27: 3, 28: 1, 29: 1, 30: 3
    },
    'modelli matematici per la biomedicina': {
        18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 3, 25: 1, 26: 0, 27: 1, 28: 5, 29: 4, 30: 13
    },
    'web applications I': {
        18: 0, 19: 0, 20: 2, 21: 1, 22: 0, 23: 1, 24: 2, 25: 3, 26: 3, 27: 4, 28: 7, 29: 10, 30: 62
    }
}

# courses to choose form table 1
grade_probabilities_table_1 = {
    'statistical methods in data science': {
        18: 15, 19: 11, 20: 6, 21: 3, 22: 3, 23: 6, 24: 2, 25: 3, 26: 5, 27: 10, 28: 8, 29: 7, 30: 17
    },
    'computational linear algebra for large scale problems': {
        18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 1, 24: 2, 25: 4, 26: 2, 27: 2, 28: 11, 29: 4, 30: 21
    }
}

# courses to choose from table 2
grade_probabilities_table_2 = {
    'decision making and optimization': {
        18: 3, 19: 2, 20: 1, 21: 0, 22: 2, 23: 0, 24: 4, 25: 3, 26: 0, 27: 4, 28: 1, 29: 3, 30: 13
    },
    'information theory and applications': {
        18: 2, 19: 0, 20: 0, 21: 0, 22: 0, 23: 1, 24: 1, 25: 0, 26: 0, 27: 3, 28: 9, 29: 5, 30: 9
    },
    'numerical optimzation for large scale problems and stochastic optimization': {
        18: 2, 19: 0, 20: 0, 21: 3, 22: 0, 23: 0, 24: 0, 25: 2, 26: 5, 27: 4, 28: 9, 29: 7, 30: 21
    },
    'object oriented programming': {
        18: 7, 19: 9, 20: 10, 21: 12, 22: 13, 23: 13, 24: 17, 25: 12, 26: 24, 27: 14, 28: 10, 29: 14, 30: 23
    }
}

# courses to choose from table 3
grade_probabilities_table_3 = {
    'computer-aided simulation lab': {
        18: 2, 19: 1, 20: 0, 21: 0, 22: 1, 23: 3, 24: 0, 25: 1, 26: 1, 27: 1, 28: 2, 29: 5, 30: 10
    },
    'network dynamics and learning': {
        18: 3, 19: 0, 20: 0, 21: 4, 22: 0, 23: 3, 24: 1, 25: 4, 26: 3, 27: 2, 28: 5, 29: 8, 30: 8
    },
    'time dependent data with markov chains': {
        18: 1, 19: 0, 20: 0, 21: 1, 22: 1, 23: 0, 24: 2, 25: 0, 26: 0, 27: 1, 28: 1, 29: 1, 30: 0
    }
}

# courses to choose from table 4
grade_probabilities_table_4 = {
    'deep natural language processing': {
        18: 5, 19: 1, 20: 0, 21: 3, 22: 1, 23: 1, 24: 0, 25: 4, 26: 2, 27: 3, 28: 8, 29: 6, 30: 21
    },
    'machine learning for iot': {
        18: 1, 19: 4, 20: 0, 21: 1, 22: 3, 23: 2, 24: 2, 25: 4, 26: 2, 27: 1, 28: 1, 29: 4, 30: 21
    }
}

# applied data science project and free ECTS courses
grade_probabilities_adsp_and_free_ECTS = {
    'applied data science project': {
        18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 1, 29: 8, 30: 15
    },
    'bioquants': {
        18: 1, 19: 0, 20: 0, 21: 2, 22: 0, 23: 0, 24: 1, 25: 0, 26: 2, 27: 2, 28: 5, 29: 5, 30: 2
    },
    'cloud computing': {
        18: 3, 19: 2, 20: 4, 21: 4, 22: 8, 23: 10, 24: 9, 25: 10, 26: 6, 27: 3, 28: 5, 29: 5, 30: 9
    },
    'computational intelligence': {
        18: 0, 19: 0, 20: 1, 21: 1, 22: 2, 23: 1, 24: 1, 25: 2, 26: 2, 27: 4, 28: 3, 29: 0, 30: 59
    },
    'financial engineering': {
        18: 0, 19: 1, 20: 1, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 1, 27: 2, 28: 0, 29: 0, 30: 0
    },
    'human computer interaction': {
        18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 2, 27: 6, 28: 26, 29: 25, 30: 88
    },
    'information system': {
        18: 1, 19: 1, 20: 3, 21: 4, 22: 5, 23: 9, 24: 7, 25: 7, 26: 8, 27: 10, 28: 8, 29: 6, 30: 15
    },
    'information system security': {
        18: 24, 19: 8, 20: 11, 21: 11, 22: 7, 23: 15, 24: 13, 25: 7, 26: 6, 27: 8, 28: 11, 29: 2, 30: 4
    },
    'metodi quantitativi per la gestione del rischio': {
        18: 0, 19: 0, 20: 1, 21: 0, 22: 2, 23: 3, 24: 2, 25: 5, 26: 3, 27: 3, 28: 1, 29: 1, 30: 3
    },
    'modelli matematici per la biomedicina': {
        18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 3, 25: 1, 26: 0, 27: 1, 28: 5, 29: 4, 30: 13
    },
    'web applications I': {
        18: 0, 19: 0, 20: 2, 21: 1, 22: 0, 23: 1, 24: 2, 25: 3, 26: 3, 27: 4, 28: 7, 29: 10, 30: 62
    }
}

################################## lists of all courses in the syllabus ##################
all_courses = ['data management and visualization', 'data science lab: process and methods',
               'data ethics and data protection', 'distributed architecture for big data processing and analytics',
               'machine learning and deep learning', 'mathematics in machine learning', 'table 1', 'table 2',
               'innovation management', 'table 3', 'table 4', 'table A', 'thesis']
courses_no_thesis = ['data management and visualization', 'data science lab: process and methods',
                     'data ethics and data protection',
                     'distributed architecture for big data processing and analytics',
                     'machine learning and deep learning', 'mathematics in machine learning', 'table 1', 'table 2',
                     'innovation management', 'table 3', 'table 4',
                     'table A']  # untill the student does not acquire 48 CFU, no thesis allowed

table_1 = ['statistical methods in data science', 'computational linear algebra for large scale problems']
table_2 = ['decision making and optimization', 'information theory and applications',
           'numerical optimzation for large scale problems and stochastic optimization', 'object oriented programming']
table_3 = ['computer-aided simulation lab', 'network dynamics and learning', 'time dependent data with markov chains']
table_4 = ['deep natural language processing', 'machine learning for iot']

table_A = ['challenge', 'internship', 'applied data science project', 'free ECTS']
table_A1 = ['challenge', 'applied data science project',
            'free ECTS']  # until the student does not acquire 48 CFU, no internship allowed

free_ECTS = ['bioquants', 'cloud computing', 'computational intelligence', 'human computer interaction',
             'financial engineering', 'information system', 'information system security',
             'metodi quantitativi per la gestione del rischio',
             'modelli matematici per la biomedicina', 'web applications I']

TOT_COURSES = len(all_courses)
NUM_SESSIONS_PER_YEAR = 4  # winter session, spring session, summer session and autumn session

# add random seed for reproducibility purposes
seed = 142
random.seed(seed)


# class Student: it contains the id, the number of left exams before graduating, all the grades achieved
# the count of the number of sessions taken to graduate, the years before graduating (one year is considered
# equal to 6 sessions), the total credits that a student got when passing exams and the relative exams,
# the final graduation grade and the list of exams that the student didn't pass
class Student():
    def __init__(self, id):
        self.id = id
        self.num_left_exams = TOT_COURSES
        # the number of left exams to pass is set to the total number of courses because when the student starts
        # his career he must have already to pass all the courses' exams
        self.grades = []
        self.sessions_to_graduate = 0
        self.years_to_graduate = 0
        self.final_grade = 0
        self.failed_exams = []
        self.bool_internship = False
        self.acquired_credits = 0
        self.acquired_cfu = []
        self.times_to_pass_last = 0


################################ other input parameters ##############################
import pandas as pd

################### name, probability of passing the exam at the first attemp,  #######
################### probability of passing the exam not at the first attemp and #######
################### associated credits for each exam in the dataframe #################

common_df = pd.DataFrame({'course': ['data management and visualization', 'data science lab: process and methods',
                                     'data ethics and data protection',
                                     'distributed architecture for big data processing and analytics',
                                     'machine learning and deep learning', 'mathematics in machine learning',
                                     'innovation management',
                                     'table 1', 'table 2', 'table 3', 'table 4', 'table A', 'bioquants',
                                     'cloud computing', 'computational intelligence', 'human computer interaction',
                                     'financial engineering', 'information system', 'information system security',
                                     'metodi quantitativi per la gestione del rischio',
                                     'modelli matematici per la biomedicina', 'web applications I', 'thesis'],
                          'probability_of_passing_first': [0.80, 0.69, 0.55, 0.68, 0.69, 0.38, 0.70, '-', '-', '-', '-',
                                                           '-', 0.71, 0.67, 0.73, 0.8, 0.17, 0.78, 0.67, 0.71, 0.6,
                                                           0.76, '-'],
                          'probability_of_passing_later': [0.74, 0.65, 0.56, 0.65, 0.67, 0.41, 0.73, '-', '-', '-', '-',
                                                           '-', 0.71, 0.6, 0.73, 0.8, 0.17, 0.70, 0.58, 0.73, 0.67,
                                                           0.67, '-'],
                          'ECTS': [8, 8, 6, 8, 10, 8, 6, 8, 8, 8, 8, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 22]
                          })
common_df_no_thesis = pd.DataFrame(
    {'course': ['data management and visualization', 'data science lab: process and methods',
                'data ethics and data protection', 'distributed architecture for big data processing and analytics',
                'machine learning and deep learning', 'mathematics in machine learning', 'innovation management',
                'table 1', 'table 2', 'table 3', 'table 4', 'table A'],
     'probability_of_passing_first': [0.80, 0.69, 0.55, 0.68, 0.69, 0.38, 0.70, '-', '-', '-', '-', '-'],
     'probability_of_passing_later': [0.74, 0.65, 0.56, 0.65, 0.67, 0.41, 0.73, '-', '-', '-', '-', '-'],
     'ECTS': [8, 8, 6, 8, 10, 8, 6, 8, 8, 8, 8, 12]
     })
table_1_df = pd.DataFrame({'course': table_1,
                           'probability_of_passing_first': [0.69, 0.76],
                           'probability_of_passing_later': [0.65, 0.75],
                           'probability_of_being_chosen': [62 / 193, 131 / 193]
                           })
table_2_df = pd.DataFrame({'course': table_2,
                           'probability_of_passing_first': [0.63, 0.69, 0.56, 0.55],
                           'probability_of_passing_later': [0.55, 0.72, 0.57, 0.46],
                           'probability_of_being_chosen': [57 / 248, 36 / 248, 85 / 248, 70 / 248]
                           })
table_3_df = pd.DataFrame({'course': table_3,
                           'probability_of_passing_first': [0.61, 0.63, 0.56],
                           'probability_of_passing_later': [0.61, 0.63, 0.56],
                           'probability_of_being_chosen': [46 / 126, 64 / 126, 16 / 126]
                           })
table_4_df = pd.DataFrame({'course': table_4,
                           'probability_of_passing_first': [0.69, 0.84],
                           'probability_of_passing_later': [0.68, 0.80],
                           'probability_of_being_chosen': [80 / 135, 55 / 135]
                           })
table_A_df = pd.DataFrame({'course': table_A,
                           'probability_of_passing_first': ['-', '-', 0.83, '-'],
                           'probability_of_passing_later': ['-', '-', 0.79, '-'],
                           'probability_of_being_chosen': [0.1, 0.444, 0.2, 0.275]
                           })
table_A1_df = pd.DataFrame({'course': table_A1,
                            'probability_of_passing_first': ['-', 0.83, '-'],
                            'probability_of_passing_later': ['-', 0.79, '-'],
                            'probability_of_being_chosen': [0.1, 0.3, 0.6]  # assumed, no statistics about it
                            })
free_ECTS_df = pd.DataFrame({'course': free_ECTS,
                             'probability_of_passing_first': [0.71, 0.67, 0.73, 0.8, 0.17, 0.78, 0.67, 0.71, 0.6, 0.76],
                             'probability_of_passing_later': [0.71, 0.6, 0.73, 0.8, 0.17, 0.70, 0.58, 0.73, 0.67, 0.67]
                             })

# initialization of the student id -> it will be updated at the end of the loop
id = 0
################ empty lists for further analyses ####################################
students = []
averages = []
sessions = []
graduation_grades = []
failures = []
max_retaken_times = []
most_difficult_courses = []
average_times_retake = []
times_for_the_last = []
who_did_internship = []
graduation_grade_who_did_internship = []
last_exam_is_thesis = []
grade_last_exam_is_thesis = []


######################## beginning of the simulation ##########ch###################
def simulate_carrier(TOT_STUDENTS):
    graduated_students = 0
    id = 0
    probability_of_passing = 0.7
    for student in range(TOT_STUDENTS):
        courses = []
        courses.extend(all_courses)
        # print(f'\ncourses to attend: {[i for i in courses]}')
        student = Student(id)
        # print(f'student {id+1}')
        # print(f'left exams: {student.num_left_exams}')
        # times_to_pass_last = 0 # how many times the student take before passing the last exam before the graduation
        total_credits = 0
        times_thesis_added_to_failed = 0
        chosen_course = None
        chosen_ECTS = None
        sessions_stuck_in_internship = None
        sessions_stuck_in_challenge = None
        sessions_stuck_in_thesis = None
        while student.num_left_exams != 0:  # the loop goes on until all the exams are passed
            for session in range(NUM_SESSIONS_PER_YEAR):
                NUM_EXAMS_PER_SESSION = random.randint(0, 5)
                if courses:
                    # print(f'\tsession {session}, year {student.years_to_graduate}')
                    # if the remaining exams are more than the maximum number of exams that a student can take in 1 session
                    # then the exams that a student takes in this session is chosen randomly among all the remaining courses
                    if student.num_left_exams > NUM_EXAMS_PER_SESSION:
                        exams_for_this_session = random.sample(courses, NUM_EXAMS_PER_SESSION)
                    # else if the number of remaining exams is less than the exams that a student can take in a session
                    # then the exams that the student takes in this session is equal to the left exams to pass
                    else:
                        exams_for_this_session = courses
                    # print(f'\t\texams for session {session}: {exams_for_this_session}')
                    # if the student take more than 3 exams in a single session, the probability of passing each exam is lowered by a certain percentage
                    if len(exams_for_this_session) >= 3:
                        minor_probability_of_passing = random.uniform(0.05, 0.15)
                        # print(f'the probabilties of passing the exams this session are lowered by {round(minor_probability_of_passing * 100, 2)}%')
                    else:
                        minor_probability_of_passing = 0.0
                    # if the student chooses to take more than 3 exams per session, then the probability of passing them is lowered by a certain percentage
                    for exam in exams_for_this_session:
                        if exam == 'table 1':
                            chosen_course = \
                            random.choices(table_1, weights=table_1_df['probability_of_being_chosen'].values, k=1)[0]
                            # print(f'\t\t\tthe student chose the course {chosen_course} from {exam}')
                            if chosen_course in student.failed_exams:  # if the student already fail/reject this exam
                                probability_of_passing = table_1_df.loc[
                                    table_1_df['course'] == chosen_course, 'probability_of_passing_later'].values[0]
                            else:  # if this is the first time trying the exam
                                probability_of_passing = table_1_df.loc[
                                    table_1_df['course'] == chosen_course, 'probability_of_passing_first'].values[0]

                        elif exam == 'table 2':
                            chosen_course = \
                            random.choices(table_2, weights=table_2_df['probability_of_being_chosen'].values, k=1)[0]
                            # print(f'\t\t\tthe student chose the course {chosen_course} from {exam}')
                            if exam in student.failed_exams:  # if the student already fail/reject this exam
                                probability_of_passing = table_2_df.loc[
                                    table_2_df['course'] == chosen_course, 'probability_of_passing_later'].values[0]
                            else:  # if this is the first time trying the exam
                                probability_of_passing = table_2_df.loc[
                                    table_2_df['course'] == chosen_course, 'probability_of_passing_first'].values[0]

                        elif exam == 'table 3':
                            chosen_course = \
                            random.choices(table_3, weights=table_3_df['probability_of_being_chosen'].values, k=1)[0]
                            # print(f'\t\t\tthe student chose the course {chosen_course} from {exam}')
                            if exam in student.failed_exams:  # if the student already fail/reject this exam
                                probability_of_passing = table_3_df.loc[
                                    table_3_df['course'] == chosen_course, 'probability_of_passing_later'].values[0]
                            else:  # if this is the first time trying the exam
                                probability_of_passing = table_3_df.loc[
                                    table_3_df['course'] == chosen_course, 'probability_of_passing_first'].values[0]

                        elif exam == 'table 4':
                            chosen_course = \
                            random.choices(table_4, weights=table_4_df['probability_of_being_chosen'].values, k=1)[0]
                            # print(f'\t\t\tthe student chose the course {chosen_course} from {exam}')
                            if exam in student.failed_exams:  # if the student already fail/reject this exam
                                probability_of_passing = table_4_df.loc[
                                    table_4_df['course'] == chosen_course, 'probability_of_passing_later'].values[0]
                            else:  # if this is the first time trying the exam
                                probability_of_passing = table_4_df.loc[
                                    table_4_df['course'] == chosen_course, 'probability_of_passing_first'].values[0]


                        elif exam == 'table A':
                            if student.acquired_credits >= 48:
                                chosen_course = \
                                random.choices(table_A, weights=table_A_df['probability_of_being_chosen'].values, k=1)[
                                    0]
                                # print(f'\t\t\tthe student chose {chosen_course} from {exam}')
                                if chosen_course == 'challenge':
                                    sessions_stuck_in_challenge = random.randint(1, 2)
                                    if student.failed_exams.count('challenge') < sessions_stuck_in_challenge:
                                        # print(f'... challenge starts ...')
                                        pass

                                    else:  # if the challenge is finished
                                        student.num_left_exams -= 1
                                        course_row = common_df[common_df['course'] == exam]
                                        # extract the ECTS
                                        cfu = course_row['ECTS'].values[0]
                                        student.acquired_credits += cfu
                                        student.acquired_cfu.append((exam, cfu))
                                        courses.remove(exam)
                                        # print(f'the challenge is finished and lasts for {sessions_stuck_in_challenge}')
                                elif chosen_course == 'internship':
                                    sessions_stuck_in_internship = random.randint(2, 5)
                                    student.bool_internship = True
                                    if student.failed_exams.count('internship') < sessions_stuck_in_internship:
                                        # print(f'... internship in progress ...')
                                        pass
                                    else:  # if the internship is finished
                                        student.num_left_exams -= 1
                                        course_row = common_df[common_df['course'] == exam]
                                        # extract the ECTS
                                        cfu = course_row['ECTS'].values[0]
                                        student.acquired_credits += cfu
                                        student.acquired_cfu.append((exam, cfu))
                                        courses.remove(exam)
                                        # print(f'the internship is finished')
                                elif chosen_course == 'applied data science project':
                                    # print(f'\t\t\tthe student chose the course {chosen_course} from {exam}')
                                    if exam in student.failed_exams:  # if the student already fail/reject this exam
                                        probability_of_passing = table_A_df.loc[table_A_df[
                                                                                    'course'] == chosen_course, 'probability_of_passing_later'].values[
                                            0]
                                    else:  # if this is the first time trying the exam
                                        probability_of_passing = table_A_df.loc[table_A_df[
                                                                                    'course'] == chosen_course, 'probability_of_passing_first'].values[
                                            0]
                                elif chosen_course == 'free ECTS':
                                    chosen_ECTS = random.sample(free_ECTS, k=2)
                                    # print(f'\t\t\tthe student chose the course {chosen_ECTS[0]} and {chosen_ECTS[1]} from {chosen_course}')
                                    courses.append(chosen_ECTS[0])
                                    courses.append(chosen_ECTS[1])
                                    student.num_left_exams += 2
                                    # if the student chooses to take free ECTS, the choices are inserted in the syllabus of the student

                            else:  # if the student did not acquire at least 48 cfu, the student cannot attend the internship, but the student can choose among the other options
                                chosen_course = \
                                random.choices(table_A1, weights=table_A1_df['probability_of_being_chosen'].values,
                                               k=1)[0]
                                # print(f'\t\t\tthe student chose {chosen_course} from {exam}')
                                if chosen_course == 'challenge':
                                    sessions_stuck_in_challenge = random.randint(1, 2)
                                    if student.failed_exams.count('challenge') < sessions_stuck_in_challenge:
                                        pass
                                        # print(f'... challenge starts ...')
                                    else:  # if the challenge is finished
                                        student.num_left_exams -= 1
                                        course_row = common_df[common_df['course'] == exam]
                                        # extract the ECTS
                                        cfu = course_row['ECTS'].values[0]
                                        student.acquired_credits += cfu
                                        student.acquired_cfu.append((exam, cfu))
                                        courses.remove(exam)
                                        # print(f'the challenge is finished and lasts for {sessions_stuck_in_challenge}')
                                elif chosen_course == 'applied data science project':
                                    # print(f'\t\t\tthe student chose the course {chosen_course} from {exam}')
                                    if exam in student.failed_exams:  # if the student already fail/reject this exam
                                        probability_of_passing = table_A_df.loc[table_A_df[
                                                                                    'course'] == chosen_course, 'probability_of_passing_later'].values[
                                            0]
                                    else:  # if this is the first time trying the exam
                                        probability_of_passing = table_A_df.loc[table_A_df[
                                                                                    'course'] == chosen_course, 'probability_of_passing_first'].values[
                                            0]
                                elif chosen_course == 'free ECTS':
                                    chosen_ECTS = random.sample(free_ECTS, k=2)
                                    # print(f'\t\t\tthe student chose the course {chosen_ECTS[0]} and {chosen_ECTS[1]} from {chosen_course}')
                                    courses.append(chosen_ECTS[0])
                                    courses.append(chosen_ECTS[1])
                                    student.num_left_exams += 2

                        elif exam == 'thesis':
                            if student.acquired_credits >= 48:
                                sessions_stuck_in_thesis = random.randint(2, 5)
                                if student.failed_exams.count('thesis') <= sessions_stuck_in_thesis:
                                    # print(f'... the student is writing the thesis ...')
                                    pass
                                else:  # if the thesis is finished
                                    student.num_left_exams -= 1
                                    courses.remove(exam)
                                    course_row = common_df[common_df['course'] == exam]
                                    # extract the ECTS
                                    cfu = course_row['ECTS'].values[0]
                                    student.acquired_credits += cfu
                                    student.acquired_cfu.append((exam, cfu))
                                    # print(f'the thesis is finished and lasts for {sessions_stuck_in_thesis} sessions')
                            else:  # if the student did not acquired 48 cfu, cannot write the thesis
                                # print(f'The student cannot choose thesis because CFU = {student.acquired_credits}')
                                break

                        elif exam in free_ECTS:
                            if chosen_ECTS != None:
                                if chosen_ECTS[
                                    0] in student.failed_exams:  # if the student already fail/reject this exam
                                    probability_of_passing = free_ECTS_df.loc[free_ECTS_df['course'] == chosen_ECTS[
                                        0], 'probability_of_passing_later'].values[0]
                                else:  # if this is the first time trying the exam
                                    probability_of_passing = free_ECTS_df.loc[free_ECTS_df['course'] == chosen_ECTS[
                                        0], 'probability_of_passing_first'].values[0]
                                if chosen_ECTS[
                                    1] in student.failed_exams:  # if the student already fail/reject this exam
                                    probability_of_passing = free_ECTS_df.loc[free_ECTS_df['course'] == chosen_ECTS[
                                        1], 'probability_of_passing_later'].values[0]
                                else:  # if this is the first time trying the exam
                                    probability_of_passing = free_ECTS_df.loc[free_ECTS_df['course'] == chosen_ECTS[
                                        1], 'probability_of_passing_first'].values[0]

                        else:  # if it is a mandatory exam, not in any of the tables
                            if exam in student.failed_exams:  # if the student already fail/reject this exam
                                probability_of_passing = \
                                common_df.loc[common_df['course'] == exam, 'probability_of_passing_later'].values[0]
                            else:  # if this is the first time trying the exam
                                probability_of_passing = \
                                common_df.loc[common_df['course'] == exam, 'probability_of_passing_first'].values[0]

                        if chosen_course != None and sessions_stuck_in_internship != None and chosen_course == 'internship' and student.failed_exams.count(
                                'internship') < sessions_stuck_in_internship:
                            student.failed_exams.append(chosen_course)
                        elif chosen_course != None and chosen_course == 'challenge' and student.failed_exams.count(
                                'challenge') < sessions_stuck_in_challenge:
                            student.failed_exams.append(chosen_course)
                        elif exam == 'thesis' and sessions_stuck_in_thesis != None and student.failed_exams.count(
                                'thesis') < sessions_stuck_in_thesis:
                            student.failed_exams.append(exam)
                        else:
                            # in all the 'ordinary' exams
                            result = random.choices(['passed', 'failed'],
                                                    [probability_of_passing - minor_probability_of_passing,
                                                     1 - probability_of_passing - minor_probability_of_passing])[0]
                            # for each taken exam, the chances to pass or fail the exam is given by the Bernoulli distribution
                            # with probability of passing fixed for each exam and tunable at the beginning of the simulation
                            if result == 'passed':
                                if exam == 'thesis' or (exam == 'table A' and (
                                        chosen_course == 'internship' or chosen_course == 'challenge')):
                                    # extreme case -> they are not treated as a common exam, so no grade is needed
                                    pass
                                else:  # for all the exams treated as they are, the student will get a grade
                                    if exam == 'table 1':
                                        grade = random.choices(list(grade_probabilities_table_1[chosen_course].keys()),
                                                               list(grade_probabilities_table_1[
                                                                        chosen_course].values()))[0]
                                    elif exam == 'table 2':
                                        grade = random.choices(list(grade_probabilities_table_2[chosen_course].keys()),
                                                               list(grade_probabilities_table_2[
                                                                        chosen_course].values()))[0]
                                    elif exam == 'table 3':
                                        grade = random.choices(list(grade_probabilities_table_3[chosen_course].keys()),
                                                               list(grade_probabilities_table_3[
                                                                        chosen_course].values()))[0]
                                    elif exam == 'table 4':
                                        grade = random.choices(list(grade_probabilities_table_4[chosen_course].keys()),
                                                               list(grade_probabilities_table_4[
                                                                        chosen_course].values()))[0]
                                    elif exam == 'table A' and chosen_course == 'applied data science project':
                                        grade = random.choices(
                                            list(grade_probabilities_adsp_and_free_ECTS[chosen_course].keys()),
                                            list(grade_probabilities_adsp_and_free_ECTS[chosen_course].values()))[0]
                                    elif exam == 'table A' and chosen_course == 'free ECTS' and chosen_ECTS[0]:
                                        grade = \
                                        random.choices(list(grade_probabilities_commmon_courses[chosen_ECTS[0]].keys()),
                                                       list(grade_probabilities_commmon_courses[
                                                                chosen_ECTS[0]].values()))[0]
                                    elif exam == 'table A' and chosen_course == 'free ECTS' and chosen_ECTS[1]:
                                        grade = \
                                        random.choices(list(grade_probabilities_commmon_courses[chosen_ECTS[1]].keys()),
                                                       list(grade_probabilities_commmon_courses[
                                                                chosen_ECTS[1]].values()))[0]


                                    else:  # if a common exam
                                        grade = random.choices(list(grade_probabilities_commmon_courses[exam].keys()),
                                                               list(
                                                                   grade_probabilities_commmon_courses[exam].values()))[
                                            0]

                                    student.grades.append(grade)

                                    # if just one course, probability to accept = 0.9
                                    if student.num_left_exams == 1:
                                        probability_of_accepting = 0.9

                                    # if the remaining courses are a few, higher probability to accept
                                    elif student.num_left_exams > 2 and student.num_left_exams <= 4:
                                        if grade > 28:
                                            probability_of_accepting = 1
                                        elif grade <= 28 and grade > 25:
                                            probability_of_accepting = 0.9
                                        elif grade <= 25 and grade > 22:
                                            probability_of_accepting = 0.8
                                        else:
                                            probability_of_accepting = 0.6

                                    # if the student already passed at least 7 exams and the average grade untill that moment is greater than 28,
                                    # higher probability to reject low grades
                                    elif len(student.grades) > 7 and statistics.mean(student.grades) > 28:
                                        if grade > 28:
                                            probability_of_accepting = 1
                                        elif grade <= 28 and grade > 25:
                                            probability_of_accepting = 0.7
                                        elif grade <= 25 and grade > 22:
                                            probability_of_accepting = 0.4
                                        else:
                                            probability_of_accepting = 0.3

                                    else:  # in all the other cases:
                                        if grade > 28:
                                            probability_of_accepting = 1
                                        elif grade <= 28 and grade > 25:
                                            probability_of_accepting = 0.8
                                        elif grade <= 25 and grade > 22:
                                            probability_of_accepting = 0.6
                                        else:
                                            probability_of_accepting = 0.5

                                    accept = random.choices(['accepted', 'rejected'],
                                                            [probability_of_accepting, 1 - probability_of_accepting])[0]
                                    # if the exam is passed, the grade is assigned based on the probability distribution shown at
                                    # the beginning of the simulation
                                    # also, the probability of accepting the grade follows a Bernoulli distribution with probability
                                    # chosen at the beginning
                                    # if the exam grade is accepted, the number of remaining exams for a student decreases by 1

                                    if accept == 'accepted':
                                        student.num_left_exams -= 1
                                        courses.remove(exam)

                                        # print(
                                        #     f'\t\t\texam {exam} is {result} with grade {grade} and it is {accept} '
                                        #     f'so now the number of left exams is {student.num_left_exams}')

                                        if exam in free_ECTS:
                                            pass
                                        else:
                                            course_row = common_df[common_df['course'] == exam]
                                            # extract the ECTS
                                            cfu = course_row['ECTS'].values[0]
                                            student.acquired_credits += cfu
                                            student.acquired_cfu.append((exam, cfu))

                                    else:  # reject
                                        # print(f'\t\t\texam {exam} is {result} with grade {grade} but it is {accept}')
                                        pass
                            else:  # failed
                                # print(f'\t\t\tthe exam {exam} is {result}, so the number of left exams is {student.num_left_exams}')
                                student.failed_exams.append(exam)
                                pass

                    if student.num_left_exams == 1:
                        student.times_to_pass_last += 1
                        if exam == 'thesis':
                            last_exam_is_thesis.append(student.id)
                            # if the number of missing exams to graduate is equal to 1, then count the number of times
                    # that the student takes to pass the last exam
                    # print(f'the remaining courses are: {courses}')

                else:  # if no courses left, stop the simulation
                    break

                student.sessions_to_graduate += 1
            student.years_to_graduate += 1  # when the number of sessions becomes equal to 4, a year is finished
            # if student.sessions_to_graduate >= 15 and np.mean(student.grades) <= 24: # if too much time to pass the exams and the mean is too low
            #   print(f'the student {student.id} quits')
            #   quitting_students += 1
            #   break
            # I decided to not implement the dropping students since this data is not present in any of the statistics under consideration

            # print(f'------- {student.sessions_to_graduate} sessions passed -------')
            # print(f'total credits acquired until now = {student.acquired_credits}')
        if student.bool_internship == True:
            # print(f'the student did internship that lasts for {sessions_stuck_in_internship} sessions')
            # remove the internship from failed exams' list
            who_did_internship.append(student.id)
            if student.id in last_exam_is_thesis:  # the bonus points are higher because the student was fully dedicated to thesis
                graduation_grade = ((sum(student.grades) / len(student.grades)) / 30) * 110 + random.uniform(2,
                                                                                                             4) + random.uniform(
                    1, 2) + random.uniform(1, 2)
            else:
                graduation_grade = ((sum(student.grades) / len(student.grades)) / 30) * 110 + random.uniform(0,
                                                                                                             4) + random.uniform(
                    0, 2) + random.uniform(0, 2)
            graduation_grade_who_did_internship.append(graduation_grade)
            for course in student.failed_exams:
                if course == 'internship':
                    student.failed_exams.remove(course)
        # else:
        # print('the student did not take the internship')

        # remove the challenge from failed exams' list
        for course in student.failed_exams:
            if course == 'challenge':
                student.failed_exams.remove(course)
        for course in student.failed_exams:
            if course == 'thesis':
                student.failed_exams.remove(course)
        # print(f'Total CFU acquired {student.acquired_credits} ')
        # print(f'Count CFU = {list((e, cfu) for e, cfu in student.acquired_cfu)}')

        ####################################### OUTPUT METRICS for each student ################################################################
        # print(f'\n\nthe student {id + 1} got the following grades: {student.grades}')

        avg_exams = sum(student.grades) / len(student.grades)
        # print(f'average grade considering {TOT_COURSES} exams: {avg_exams}')
        # print(f'number of sessions to graduate: {student.sessions_to_graduate}')
        # floor_years = math.floor(student.sessions_to_graduate/NUM_SESSIONS_PER_YEAR)
        # print(f'number of years to graduate: {floor_years} years and {student.sessions_to_graduate - (floor_years * NUM_SESSIONS_PER_YEAR)} sessions')
        # print(f'number of failed exams: {len(student.failed_exams)}')
        # print(
        #     f'most retaken exam before passing it:  Course {statistics.mode(student.failed_exams)} -> Course {statistics.mode(student.failed_exams)} was failed {max(Counter(student.failed_exams).values())} times')
        # print(
        #     f'average of the times the student retook an exam: {statistics.mean(Counter(student.failed_exams).values())} times')
        # print(f'the last exams is passed at the attempt number {student.times_to_pass_last}')

        if student.id in last_exam_is_thesis:  # the bonus points are higher because the student was fully dedicated to thesis
            graduation_grade = (avg_exams / 30) * 110 + random.uniform(2, 4) + random.uniform(1, 2) + random.uniform(1,
                                                                                                                     2)
            grade_last_exam_is_thesis.append(graduation_grade)
        else:
            graduation_grade = (avg_exams / 30) * 110 + random.uniform(0, 4) + random.uniform(0, 2) + random.uniform(0,
                                                                                                                     2)
        graduated_students += 1
        # note: the assignment of the bonus points is uniformly distributed
        if graduation_grade > 112.5:
            # print(f'the graduation grade of student {student.id + 1} is 110 cum laude')
            graduation_grades.append('110 cum laude')
        else:
            # print(f'the graduation grade of student {student.id + 1} is {graduation_grade}')
            graduation_grades.append(graduation_grade)
        # print('\n\n\nfinish')
        ############################################ lists are fulfilled for further analyses #################################
        students.append(student.id)
        averages.append(avg_exams)
        sessions.append(student.sessions_to_graduate)
        failures.append(len(student.failed_exams))
        max_retaken_times.append(max(Counter(student.failed_exams).values()))
        most_difficult_courses.append(statistics.mode(student.failed_exams))
        average_times_retake.append(statistics.mean(Counter(student.failed_exams).values()))
        times_for_the_last.append(student.times_to_pass_last)

        # the identification number of each student is updated at the end of the loop
        id += 1
    print(f'the global mean of the graduation grade is {np.mean(graduation_grades)}')
    print(
        f'{len(who_did_internship)} over {TOT_STUDENTS} students chose to attend an internship -> their graduation grade is {np.mean(graduation_grade_who_did_internship)}')
    graduation_grade_who_did_internship.clear()
    who_did_internship.clear()
    print(
        f'the number of students who were fully dedicated to the thesis is {len(set(last_exam_is_thesis))} over {TOT_STUDENTS} and their mean graduation grade is {sum(set(grade_last_exam_is_thesis)) / len(set(grade_last_exam_is_thesis))}\n')
    grade_last_exam_is_thesis.clear()
    last_exam_is_thesis.clear()
    return graduation_grades, sessions, students, averages, failures, max_retaken_times, most_difficult_courses, average_times_retake, times_for_the_last


from scipy.stats import t


def calculate_confidence_interval(data, conf):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / (len(data) ** (1 / 2))  # this is the standard error

    interval = t.interval(confidence=conf,  # confidence level
                          df=max(len(graduation_grades) - 1, 1),  # degree of freedom
                          loc=mean,  # center of the distribution
                          scale=se  # spread of the distribution
                          # (we use the standard error as we use the extimate of the mean)
                          )

    MOE = interval[1] - interval[0]  # this is the margin of error
    re = (MOE / (2 * abs(mean)))  # this is the relative error
    acc = 1 - re  # this is the accuracy
    return interval, acc


# Confidence level
CONFIDENCE_LEVEL = .90
# Acceptance for the accurace
ACC_ACCEPTANCE = .97
studs_batch = 4
acc_grades = 0.0
acc_periods = 0.0
############################ empty lists for further analyses ####################################
last_students = []
last_averages = []
last_sessions = []
last_graduation_grades = []
last_failures = []
last_max_retaken_times = []
last_most_difficult_courses = []
last_average_times_retake = []
last_times_for_the_last = []
# dropping_students=[]
while acc_grades < ACC_ACCEPTANCE or acc_periods < ACC_ACCEPTANCE:
    # for computing the confidence intervals, I take different batches to get the higher accuracy with a fixed confidence level (0.97)
    for batch in range(studs_batch):
        print(f'Batch {batch}:')
        TOT_STUDENTS = 100  # each batch contains 100 students and computes the confidence intervals and teh accuracy taking care of all of them
        graduation_grades, sessions, students, averages, failures, max_retaken_times, most_difficult_courses, average_times_retake, times_for_the_last = simulate_carrier(
            TOT_STUDENTS)
        # for index in students:
        #   if sessions[index] >= 17 and averages[index] <= 24:
        #     # if the student takes more than 4 year to graduate and the average grade is too low, do not count
        # in the end, I chose not to implement the dropping students since they are not reported in the AlmaLaurea website
        #     dropping_students.append(index)
        # for index in dropping_students:
        #     graduation_grades.pop(index)
        #     sessions.pop(index)
        #     students.pop(index)
        #     averages.pop(index)
        #     failures.pop(index)
        #     max_retaken_times.pop(index)
        #     most_difficult_courses.pop(index)
        #     average_times_retake.pop(index)
        #     times_for_the_last.pop(index)
        interval_1, acc_grades = calculate_confidence_interval(graduation_grades, CONFIDENCE_LEVEL)
        interval_2, acc_periods = calculate_confidence_interval(sessions, CONFIDENCE_LEVEL)
        last_students = students[len(students) - TOT_STUDENTS:]
        last_averages = averages[len(students) - TOT_STUDENTS:]
        last_sessions = sessions[len(students) - TOT_STUDENTS:]
        last_graduation_grades = graduation_grades[len(students) - TOT_STUDENTS:]
        last_failures = failures[len(students) - TOT_STUDENTS:]
        last_max_retaken_times = max_retaken_times[len(students) - TOT_STUDENTS:]
        last_most_difficult_courses = most_difficult_courses[len(students) - TOT_STUDENTS:]
        last_average_times_retake = average_times_retake[len(students) - TOT_STUDENTS:]
        last_times_for_the_last = times_for_the_last[len(students) - TOT_STUDENTS:]
        # print(f'\t\t\t\t\t FINE BATCH {batch}')

print(f'\nCOMPUTATION OF CI FINISHED WITH ACC_GRADES = {acc_grades} AND ACC_PERIODS = {acc_periods}\n')

######################################## ANALYSES PART #################################################################
########################### confidence level, it can be tuned #########################################################

import matplotlib.pyplot as plt
import scipy.stats as stats

plt.figure(figsize=(22, 28))

##################################### graduation grades plot + confidence interval #####################################

print(f"Confidence Interval for graduation grades: [{interval_1[0]}, {interval_1[1]}]")
print(f'The accuracy for graduation grades is {acc_grades}')
print(
    f'The maximum graduation grade is {max(last_graduation_grades)} while the minimum graduation grade is {min(last_graduation_grades)}')
print(f'The mean of the graduation grades is {np.mean(last_graduation_grades)} considering {TOT_STUDENTS} students')

plt.subplot(7, 1, 1)
plt.plot(last_students, last_graduation_grades, color='blue', label='Graduation Grades')
plt.title('Graduation grades for each student')
plt.xlabel('Student ID')
plt.ylabel('Graduation Grade')
plt.axhline(y=interval_1[0], color='red', linestyle='--', label='CI Lower Bound')
plt.axhline(y=interval_1[1], color='red', linestyle='--', label='CI Upper Bound')
plt.axhline(y=np.mean(last_graduation_grades), color='brown', label='Mean grade')
plt.legend(loc='upper right')
plt.xticks(last_students)  # Set x-axis ticks to show every student ids
plt.yticks(range(int(min(last_graduation_grades)), int(max(last_graduation_grades) + 1)))
plt.grid(True)

######################################### plot graduation grades distribution ##########################################
grade_distribution = []

for grade in last_graduation_grades:
    grade_distribution.append(round(grade))

plt.subplot(7, 1, 2)
plt.hist(grade_distribution, bins=sorted(set(grade_distribution)), color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Grade Distribution')
plt.xlabel('Grade (rouded to the nearest one)')
plt.ylabel('Number of students')
plt.xticks(sorted(set(grade_distribution)))
plt.grid(True)

########################################### average grades plot ##################################

# print(f"\nConfidence Interval for average grades: [{avg_confidence_interval_lower}, {avg_confidence_interval_upper}]")
# print(f'The accuracy for average grades is {avg_accuracy}')
print(f'The maximum average grade is {max(last_averages)} while the minimum average grade is {min(last_averages)}')
print(f'The mean of the average grades is {np.mean(last_averages)}')

plt.subplot(7, 1, 3)
plt.plot(last_students, last_averages, color='green', label='Averages')
plt.xlabel('Student ID')
plt.ylabel('Average Exam Grade')
plt.title('Average grade for each student')
plt.axhline(y=np.mean(last_averages), color='brown', label='Mean average')
plt.legend(loc='upper right')
plt.xticks(last_students)
plt.grid(True)

################################### number of sessions plot + confidence interval ######################################

print(f"\nConfidence Interval for the number of sessions: [{interval_2[0]}, {interval_2[1]}]")
print(f'The accuracy for the number of sessions is {acc_periods}')
print(
    f'The maximum number of sessions taken to graduate is {max(last_sessions)} while the minimum number of sessions to graduate is {min(last_sessions)}')
print(f'The mean number of sessions taken to graduate is {np.mean(last_sessions)}')

plt.subplot(7, 1, 4)
plt.plot(last_students, last_sessions, color='orange', label='Number of Sessions')
plt.xlabel('Student ID')
plt.ylabel('Number of Sessions to Graduate')
plt.title('Session to graduate for each student')
plt.axhline(y=interval_2[0], color='red', linestyle='--', label='CI Lower Bound')
plt.axhline(y=interval_2[1], color='red', linestyle='--', label='CI Upper Bound')
plt.axhline(y=np.mean(last_sessions), color='brown', label='Mean number of sessions')
plt.legend(loc='upper right')
plt.xticks(last_students)
plt.grid(True)

######################################### plot graduation time distribution ##########################################
session_distribution = []

for grade in last_sessions:
    session_distribution.append(round(grade))

plt.subplot(7, 1, 5)
plt.hist(session_distribution, bins=sorted(set(session_distribution)), color='hotpink', edgecolor='black', alpha=0.7)
plt.title('Session Distribution')
plt.xlabel('Number of sessions to graduate (rouded to the nearest one)')
plt.ylabel('Number of students')
plt.xticks(sorted(set(session_distribution)))
plt.grid(True)

##################### max number of attempts for an exam plot ####################################
# it counts how many times the student fails an exam and must retake it -> the maximum number of times that a student
# must retake an exam is stored and plotted here

print(
    f'\nThe maximum number of attempts of retaking an exam is {max(last_max_retaken_times)} while the minimum number of attempts of retaking an exam is {min(last_max_retaken_times)}\n')

plt.subplot(7, 1, 6)
plt.plot(last_students, last_max_retaken_times, color='black', label='Max number of times the student retake an exam')
plt.xlabel('Student ID')
plt.ylabel('Times to retake an exam')
plt.title('Maximum number of attempts to retake an exam')
plt.legend(loc='upper right')
plt.xticks(last_students)
plt.grid(True)

############################################## print the most difficult courses ########################################
# for each student it is computed the number of times that he retake the exams -> the exam associated with the maximum
# number of times that the student fails the exam is named as the most difficult exam to pass for him
plt.subplot(7, 1, 7)
plt.scatter(last_students, last_most_difficult_courses, color='purple', label='Most difficult exam to pass')
plt.ylabel('Most difficult exam to pass')
plt.xlabel('Student ID')
plt.title('Most difficult exams to pass')
plt.legend(loc='upper right')
plt.xticks(last_students)
plt.yticks(range(0, 16))
plt.grid(True)

plt.tight_layout()
plt.show()

print('\n')
######################################## count what are the most difficult courses to pass #############################
# to compute this metric, the idea is to count how many students retake a specific exam
occurrences = Counter(last_most_difficult_courses)
ordered_occurrences = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)
for course in ordered_occurrences:
    print(f'The course {course[0]} is the most difficult for {course[1]} students')

########################## number of times retaking the last exam before passing it ####################################
# it computes the number of times that a student takes doing the last exam (i.e., student.num_left_courses == 1) before
# finishing all the exams to attend and graduating
print(f'\nThe number of times retaking the last exam before graduating is {max(last_times_for_the_last)} '
      f'while the minimum is {min(last_times_for_the_last)}\n')

############################################ FURTHER INTERESTING ANALYSES ##############################################
import pandas as pd

# create a dataframe to better understand
students_serie = pd.Series(last_students, name='Student id')
sessions_serie = pd.Series(last_sessions, name='Sessions to graduate')
graduation_grades_serie = pd.Series(last_graduation_grades, name='Graduation grade')
average_times_retake_serie = pd.Series(last_average_times_retake, name='Average times retaking an exam')
times_for_the_last_serie = pd.Series(last_times_for_the_last, name='Times to pass the last exam')

df = pd.DataFrame(
    {'Student id': students_serie, 'Sessions to graduate': sessions_serie, 'Graduation grade': graduation_grades_serie,
     'Average times retaking an exam': average_times_retake_serie,
     'Times to pass the last exam': times_for_the_last_serie})
df.set_index('Student id', inplace=True)
print(df.describe())
# this method allows to display some statistics about the data, like the mean, the standard deviation, the minimum, the
# maximum and the quantile (25%, 50%, 75%) for each considered attribute of teh dataframe

# additional analyses can be done comparing and filtering the students in the dataframe
# for example, it can be computed the most talented students by filtering the student ids who
# - took at most the minimum number of sessions plus 2 sessions
# - got the average times to retake an exam less than 1 time and a half
# - took the minimum number of times for passing the last exam
best_students = df[(df['Sessions to graduate'] <= min(last_sessions) + 2)
                   & (df['Average times retaking an exam'] <= 1.5)
                   & (df['Times to pass the last exam'] == min(last_times_for_the_last))]
print(f'the best students are: {best_students}')

