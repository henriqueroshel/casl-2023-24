import re        # search for regular expressions
import requests  # access website with data sources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb
from scipy import stats
from collections import namedtuple
from tqdm import tqdm   # progress bar

MIN_GRADE, MAX_GRADE = 18,30

# implementation of a Course collecting data from 
# Polito website given the url or the course code
class Course:
    def __init__(self, url_examdata=None, code=None):

        if all([url_examdata, code]) and (code not in url_examdata):
            raise TypeError("url_examdata and course_code do not match")
        
        # get data from Polito website
        approval_prob, grades_count = get_exam_data(url_examdata=url_examdata, course_code=code)

        self.approval_prob = approval_prob
        self.code = code if code else re.findall('[0-9]{2}[A-Z]{5}', url_examdata)[0]
        
        # dict with cumulative probability for each possible grade
        self.grades_cumulat_dist = dict()
        exams_passed = np.sum(grades_count)
        cumulat_prob = 0
        for i,grd in enumerate( range(MIN_GRADE,MAX_GRADE+1) ):
            cumulat_prob += grades_count[i] / exams_passed
            self.grades_cumulat_dist[grd] = cumulat_prob
    def __repr__(self):
        return f'{self.code} - Approval prob: {self.approval_prob:.2%}'

    def exam_success(self):
        # pass or not pass an exam from the course according to a
        # bernoulli experiment with parameter p=self.approval_prob
        return np.random.uniform(0,1) < self.approval_prob    
    def grade_generator(self):
        # generate a grade, based on the cumulative distribution
        u = np.random.uniform(0,1)
        for grd in range(MIN_GRADE,MAX_GRADE+1):
            if u <= self.grades_cumulat_dist[grd]:
                return grd

# Data collection from Polito website "Statistiche superamento esami"
def get_exam_data(url_examdata=None, course_code=None):
    """
    Get exam results data from Polito webpage url_examdata or from course_code
    Args:
        url_examdata (str, optional): url to Polito course webpage with 
        "Statistiche superamento esami".
        course_code (str, optional): code of the course.
    Returns:
        approval_prob, grades_count (tuple): data extracted from the url
    """    
    if course_code and not url_examdata:
        # fit the code to the base url
        base_url = 'https://didattica.polito.it/pls/portal30/esami.superi.grafico?p_cod_ins='
        url_examdata = base_url+course_code

    # get html content
    url = requests.get(url_examdata)
    htmltext = url.text
    lines = htmltext.split('\n')

    if re.search('Nessun dato per questo codice', htmltext):
        if course_code:
            raise Exception(f'No data available on the course {course_code}')
        raise Exception(f'No data available on page {url_examdata}')
        
    # use regular expressions to search specific lines in the page
    for i,line in enumerate(lines):
        if re.search('Totale iscritti: [0-9]+', line):
            # gets approval_prob on the considered exam session
            exams_taken = re.search('Totale iscritti: [0-9]+ ', line).group()
            exams_passed = re.search('Superi: [0-9]+ ', line).group()
            exams_taken = int( re.search('[0-9]+', exams_taken).group() )
            exams_passed = int( re.search('[0-9]+', exams_passed).group() )
            approval_prob = exams_passed / exams_taken
        elif re.search("name: 'Iscritti',", line):
            # grades distribution from 15 to 30 is on the next line
            grades_count = np.array( re.findall('[0-9]+', lines[i+1]), dtype=np.int32 )
            # remove count of 15-17 that are always zero
            grades_count = grades_count[3:]
            
            # for small samples with counts equal to 0, add 1 to each grd_count,
            # so that all grades are possible, maintaining the same approval_prob
            if len( grades_count[ grades_count==0 ] ) > 0:
                grades_count += 1

    return approval_prob, grades_count

def get_career_data(courses_list, method='code'):
    """
    Instantiate Course objects from a list a courses
    Args:
        courses_list (list, str): list of courses, given as a list
            or as a path to a text file
        method (['code', 'urls']) : method of collection of data;
            'code': courses_list contains a list of course code
                    to be inserted on a base url to exams data
            'url': courses_list contains a list of urls to courses 
                    pages with exams data
    Returns:
        data (list): list with instances of Course objects
    """    
    data = []
    option = 0
    option_desc = ['list with course codes', 'textfile with courses codes', 'textfile with courses urls']

    if isinstance(courses_list,str): # textfile path case 
        with open(courses_list, 'r') as f:
            courses_list = f.read().split('\n') # converts to list
            option += 1 # match the description list

    if not isinstance(courses_list, list):
        raise TypeError('courses_list can be either a list or a textfile path')

    if method=='code':
        courses_list = tqdm(courses_list, desc=f'data collected from {option_desc[option]}', ncols=123)
        for code in courses_list:
            data.append( Course( code=code ) )
    elif method=='url':
        option += 1 # match the description list
        courses_list = tqdm(courses_list, desc=f'data collected from {option_desc[option]}', ncols=123)
        for url in courses_list:
            data.append( Course( url_examdata=url ) )
    else:
        raise NameError(f'Method {method} not specified for collecting data')
    return data

def binomial_pmf(k,n,p):
    # computes p(X=k) for X \sim Bin(n,p)
    return comb(n,k) * p**k * (1-p)**(n-k)
def exams_per_session_gen(exams_left, avg_exams_per_session):
    """
    Generate the number of exams a student takes in a session following a 
    binomial distribution; the decision of taking each exam follows a bernoulli 
    distribution with probability equals to 0.5 (exactly, or the closest value 
    such that the parameter n of the binomial distribution is a integer)
    """
    # p is the closest value to 0.5 such that n is integer
    n = round( avg_exams_per_session / 0.5 )
    p = avg_exams_per_session / n

    # draw a value from a binomial random variable with parameter n and p
    # equivalent to np.random.binomial(n,p)
    k = -1
    cumulative_p=0
    u = np.random.uniform(0,1)
    while cumulative_p < u:
        k += 1
        cumulative_p += binomial_pmf(k, n, p)
    exams_draw = k    
    
    # if the drawn value is greater than the remaining exams
    # the student takes all the exams left
    return min(exams_left, exams_draw)

def reject_mark(mark, rejection_50p):
    # to handle the rejection of a mark, we consider a bernoulli experiment
    # where the parameter p=reject_prob is the probability of rejecting the mark.
    # p depends on the mark, following a sigmoid p(x)=1/(1+exp(x-rejection_50p)))
    if mark == MAX_GRADE: # not rejectable
        return False
    reject_prob = 1 / (1 + np.exp(mark-rejection_50p) )
    u = np.random.uniform(0,1) 
    return u<reject_prob

def confidence_interval(values, conf_level=0.98):
    # computes the confidence interval of a particular 
    # measure where values is a list of empirical values
    n = len(values)                 # number samples
    avg = np.mean(values)           # sample mean
    std = np.std(values, ddof=1)    # sample standard deviation
    
    if n<30: # t distribution
        ci_lower,ci_upper = stats.t.interval(conf_level, df=n-1, loc=avg, scale=std/n**.5)
    else:# normal distribution
        ci_lower,ci_upper = stats.norm.interval(conf_level, loc=avg, scale=std/n**.5)
    delta = (ci_upper-ci_lower) / 2
    return avg, delta

def accuracy(value, delta):
    # computes the accuracy of a measure given its value
    # and the semi-width (delta) of the confidence interval    
    eps = delta / value # relative error
    acc = 1 - eps # accuracy
    return max(acc, 0) # return only non-negative values

# auxiliar object for the simulation
CourseTaken = namedtuple('CourseTaken', ['course','semester','passed','grade'])
# semester allows the student to take the second exam appeal during one specific session of the year

def simulation(courses, avg_exams_per_session, sessions_per_year, rejection_50p, courses_semester=None):
    # perform a simulation of the graduation of a single student
    # courses (array) - data on the courses the student needs to pass in order to graduate
    # courses_semester (array) - 0 for courses on the first semester, 1 for second semester
    # (default alternating courses)

    total_courses = len(courses)
    exams_left = total_courses
    n_sessions = 0
    
    if not courses_semester:
        courses_semester = [i%2 for i in range(total_courses)]
    elif len(courses_semester) != total_courses:
        raise ValueError(f'Length of courses ({total_courses}) and courses_semester ({len(courses_semester)}) do not match')


    graduation_log = [None] * total_courses
    for i in range(total_courses):
        if courses_semester:
            sem = courses_semester[i] if courses_semester else i%2
        graduation_log[i] = CourseTaken(courses[i], sem, 0, 0)

    # loop until there are no pending exams to pass
    while exams_left > 0:
        # exam session - stochastic elements
        # number of exams taken during the session as a binomial random variable
        n_exams_taken = exams_per_session_gen(exams_left, avg_exams_per_session)
        # exams passed and accepted during the session
        session_passed = 0

        for i in range(n_exams_taken):
            # takes an exam
            course = graduation_log[i].course
            sem = graduation_log[i].semester
            grade = course.grade_generator() # if pass, otherwise ignored
            if course.exam_success() and not reject_mark(grade, rejection_50p):
                # exam passed and mark accepted
                graduation_log[i] = CourseTaken(course, sem, 1, grade)
                session_passed += 1
            # second exam appeal - once a year each exam is offered twice in a session
            elif (n_sessions - sem) % sessions_per_year == 0:
                grade_2appeal = course.grade_generator() # if pass, otherwise ignored
                if course.exam_success() and not reject_mark(grade_2appeal, rejection_50p):
                    # exam passed and mark accepted
                    graduation_log[i] = CourseTaken(course, sem, 1, grade_2appeal)
                    session_passed += 1
        
        exams_left -= session_passed
        # sort so the remaining courses are on top
        graduation_log = sorted(graduation_log, key=lambda k:k.passed)
        n_sessions += 1

    grades = np.array([ mark.grade for mark in graduation_log])
    final_grade = grades.mean()
    graduation_time = n_sessions / sessions_per_year
    return final_grade, graduation_time

def n_simulations(courses, avg_exams_per_session, sessions_per_year, rejection_50p, 
                  conf_level, min_accuracy, courses_semester=None):
    # perform simulations until the minimum accuracy is met for both metrics 
    total_courses = len(courses)
    grade_nsims = 0
    time_nsims = 0
    acc_grade, acc_time = 0, 0
    # get the values of the first simulation
    final_grade, graduation_time = simulation( courses, 
                                               avg_exams_per_session, 
                                               sessions_per_year,
                                               rejection_50p,
                                               courses_semester )
    n_simulations = 1
    grades_list = [ final_grade ]
    gradtimes_list = [ graduation_time ]

    while min(acc_grade, acc_time) < min_accuracy:
        final_grade, graduation_time = simulation( courses, 
                                                   avg_exams_per_session, 
                                                   sessions_per_year,
                                                   rejection_50p,
                                                   courses_semester )
        n_simulations += 1
        grades_list.append(final_grade)
        gradtimes_list.append(graduation_time)

        # confidence interval and accuracy computation
        avg_grade, delta_grade = confidence_interval(grades_list, conf_level)
        avg_time, delta_time = confidence_interval(gradtimes_list, conf_level)
        acc_grade = accuracy(avg_grade, delta_grade)
        acc_time = accuracy(avg_time, delta_time)

    return (avg_grade, delta_grade), (avg_time, delta_time)

def plot_graduation_time(parameter_values, graduation_time_values, 
                         parameter_name, save_dir='.\\figures', save_suffix=''):
    # plot parameter_values and the respective graduation time
    fig,ax = plt.subplots()
    ax.plot(parameter_values, graduation_time_values, '.', markersize=12)
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('Years')
    ax.grid()
    try:
        fig.savefig(f'{save_dir}\\g5_gradtime_{save_suffix}.png', format='png')
    except:
        fig.savefig(f'.\\g5_gradtime_{save_suffix}.png', format='png')

def plot_grade_and_time(parameter_values, graduation_time_values, grade_values, 
                        parameter_name, save_dir='.\\figures', save_suffix=''):
    fig,ax = plt.subplots(2, sharex=True, figsize=(6.4, 7.5))
    ax[0].plot(parameter_values, graduation_time_values, '.', markersize=12)
    ax[1].plot(parameter_values, grade_values, '.', markersize=12)
    ax[0].set_ylabel('Years')
    ax[1].set_ylabel('Final grade')
    ax[1].set_xlabel(parameter_name)
    ax[0].grid()
    ax[1].grid()
    try:
        fig.savefig(f'{save_dir}\\g5_grade_time_{save_suffix}', format='png')
    except:
        fig.savefig(f'.\\g5_grade_time_{save_suffix}.png', format='png')


if __name__ == '__main__':
    seed = 4028922
    np.random.seed(seed)
    CONFIDENCE_LEVEL = 0.98

    print('--- DATA COLLECTION ---')
    # option 1 - LIST WITH COURSES CODES
    # example: possible courses taken by a student 
    # to graduate on MSc Data Science and Engineering
    courses_code = [ 
        '01TXASM','01TWZSM','01URZSM','01TUYSM','01TXFSM','01TXGSM','01TXISM',
        '01TWWSM','01DTGSM','01DTHSM','01TXPSM','01URROV','01DTUNG'
    ]
    # option 2 - TEXTFILE WITH COURSES URLS
    # example: possible courses taken by a student to graduate 
    # on MSc Automotive Engineering (Industrial Processes pathway)
    urls_filename = '.\\sourcedata\\02_urls.txt'
    # option 3 - TEXTFILE WITH COURSES CODES
    # example: possible courses taken  by a student to graduate 
    # on MSc Biomedical Engineering (Bionanotechnologies pathway)
    courses_code_filename = '.\\sourcedata\\03_codes.txt' 

    coursesdata1 = get_career_data(courses_code, method='code')
    coursesdata2 = get_career_data(urls_filename, method='url')
    coursesdata3 = get_career_data(courses_code_filename, method='code')

    # other input parameters
    courses_semester = [0,0,0,0,1,1,1,1,0,0,0,1,1] # example for the considered careers with 13 courses
    min_accuracy = 0.98
    sessions_per_year = 3

    print('\n--- SELECT AN EXPERIMENT: ---')
    print('1 - validation - results for a single set of input parameters')
    print('2 - graduation time and final grade vs. rejection threshold')
    print('3 - graduation time vs. average exams taken by exam session')
    experiment = input()
    
    if experiment == '1':
        print("\n--- EXPERIMENT 1: validation - single results ---")
        # Input parameters
        avg_exams_per_session = 4.0
        # grade to which the chance of rejection is 50%
        rejection_50p = int(input('Rejection threshold (50% chance of rejecting this mark): '))

        for career in [coursesdata1, coursesdata2, coursesdata3]:
            print('INPUT PARAMETERS:')
            print(f'Courses taken to graduate: {[c.code for c in career]}')
            print(f'Average exams taken at each session: {avg_exams_per_session}')
            print(f'Exam sessions per year: {sessions_per_year}')
            print(f'Minimum accuracy of output measures: {min_accuracy}')

            grade,graduation_time = n_simulations(
                                        courses=career,
                                        avg_exams_per_session=avg_exams_per_session, 
                                        sessions_per_year=sessions_per_year,
                                        rejection_50p=rejection_50p,
                                        min_accuracy=min_accuracy, 
                                        conf_level=CONFIDENCE_LEVEL, 
                                        courses_semester=courses_semester
                                    )
            print('OUTPUT MEASURES')
            print(f'Average graduation grade: {grade[0]:.2f} \u00b1 {grade[1]:.2f}')
            print(f'Average graduation time: {graduation_time[0]:.2f} \u00b1 {graduation_time[1]:.2f} years\n')

    elif experiment == '2':
        print('\n--- EXPERIMENT 2: graduation time and final grade vs. rejection threshold ---')

        print('SELECT AN MASTER\'S DEGREE:')
        print('1 - MSc Data Science and Engineering')
        print('2 - MSc Automotive Engineering (Industrial Processes pathway)')
        print('3 - MSc Biomedical Engineering (Bionanotechnologies pathway)')
        selection = int(input())-1
        career = [coursesdata1, coursesdata2, coursesdata3][ selection ]

        # Input parameters
        avg_exams_per_session = 3.5

        print('INPUT PARAMETERS:')
        print(f'Courses taken to graduate: {[c.code for c in career]}')
        print(f'Exam sessions per year: {sessions_per_year}')
        print(f'Minimum accuracy of output measures: {min_accuracy}\n')

        # considered values
        rejection50p_list = np.arange(18,29)
        
        n_values = len(rejection50p_list)
        gradtime_avg = np.zeros( n_values )
        gradtime_delta = np.zeros( n_values )
        finalgrade_avg = np.zeros( n_values )
        finalgrade_delta = np.zeros( n_values )

        for i,rejection50p in enumerate(tqdm(rejection50p_list)):
            # simulate each value until minimum accuracy is met
            grade, graduation_time = n_simulations(
                courses=career,
                avg_exams_per_session=avg_exams_per_session, 
                sessions_per_year=sessions_per_year,
                rejection_50p=rejection50p,
                min_accuracy=min_accuracy, 
                conf_level=CONFIDENCE_LEVEL, 
                courses_semester=courses_semester
                )
            gradtime_avg[i], gradtime_delta[i] = graduation_time
            finalgrade_avg[i], finalgrade_delta[i] = grade
        

        plot_grade_and_time(
            rejection50p_list, 
            gradtime_avg, 
            finalgrade_avg, 
            'Rejection threshold',
            save_suffix=['data','auto','biom'][selection]
        )

    elif experiment == '3':
        print('\n--- EXPERIMENT 3: graduation time vs. average exams taken by exam session ---')

        print('SELECT AN MASTER\'S DEGREE:')
        print('1 - MSc Data Science and Engineering')
        print('2 - MSc Automotive Engineering (Industrial Processes pathway)')
        print('3 - MSc Biomedical Engineering (Bionanotechnologies pathway)')
        selection = int(input())-1
        career = [coursesdata1, coursesdata2, coursesdata3][ selection ]

        # Input parameters
        min_accuracy = 0.98
        sessions_per_year = 3
        rejection_50p = 24  # grade to which the chance of rejection is 50%

        print('INPUT PARAMETERS:')
        print(f'Courses taken to graduate: {[c.code for c in career]}')
        print(f'Rejection threshold (50% chance of rejecting for this mark): {rejection_50p}')
        print(f'Exam sessions per year: {sessions_per_year}')
        print(f'Minimum accuracy of output measures: {min_accuracy}\n')

        # considered values
        avg_exams_per_session_list = np.arange(2,5.1,0.25)
        
        n_values = len(avg_exams_per_session_list)
        gradtime_avg = np.zeros( n_values )
        gradtime_delta = np.zeros( n_values )
        finalgrade_avg = np.zeros( n_values )
        finalgrade_delta = np.zeros( n_values )

        for i,avg_exams_per_session in enumerate(tqdm(avg_exams_per_session_list)):
            # simulate each value until minimum accuracy is met
            grade, graduation_time = n_simulations(
                courses=career,
                avg_exams_per_session=avg_exams_per_session, 
                sessions_per_year=sessions_per_year,
                rejection_50p=rejection_50p,
                min_accuracy=min_accuracy, 
                conf_level=CONFIDENCE_LEVEL, 
                courses_semester=courses_semester
                )
            gradtime_avg[i], gradtime_delta[i] = graduation_time
            finalgrade_avg[i], finalgrade_delta[i] = grade
        
        plot_graduation_time(
            avg_exams_per_session_list, 
            gradtime_avg, 
            'Average exams taken by session',
            save_suffix=['data','auto','biom'][selection]
        )
