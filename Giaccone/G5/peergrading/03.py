########################################################
#
#   THIS SCRIPT IS TO SHOW THE FUNCTIONING OF THE SIMULATOR,
#   AND HOW THE METRICS ARE RETURNED, IT DOES NOT PRODUCE
#   ALL PLOTS SHOWN IN THE REPORT, ONLY THE ONES ASKED
#   FOR IN THE PADLET FROM LAB G4 AND THAT WERE DESCRIBED
#   IN THE REPORT IN THE SECTION "OUTPUT METRICS"
#
#######################################################

import numpy as np
from collections import defaultdict
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binom

class Data:
    """
    Class used to store every data value needed to compute the output metrics
    and also to compute the confidence intervals on the averages

    """
    def __init__(self):
        # number of students who graduated
        self.n_graduated_student = 0
        # sum of the final grades
        self.sum_final_grades = 0
        # sum of the final grades squared (used for variance computations)
        self.sum_final_grades_squared = 0
        # sum of the graduation times
        self.sum_graduation_times = 0
        # sum of the graduation times squared (used for variance computations)
        self.sum_graduation_times_squared = 0

        # dictionary storing how many student graduated with each final grade,
        # that is, key = grade, value = count of students
        self.final_grade_histogram = defaultdict(int)

        # sum of number of sessions taken to graduate
        self.sum_sessions_taken = 0
        # sum of number of sessions squared taken to graduate (used for variance computations)
        self.sum_sessions_taken_squared = 0

        # dictionary storing how many student graduated with each graduation time,
        # that is, key = graduation time, value = count of students
        self.graduation_time_histogram = defaultdict(int)

        # dictionary where each key is the graduation time, the value is a tuple where
        # the first value is the sum of the final grades of the students that graduated in that
        # (key) time, and the second value is the number of students who graduated in that
        # (key) time
        self.grade_by_graduation_time_counts = defaultdict(lambda: [0, 0])

        # number of student that could not graduate in time
        self.non_grad_student = 0


# returns the variance having just the number of samples, sum of the samples, and sum of the samples squared
def var(n_samples, sum_samples, sum_samples_squared):
    return (sum_samples_squared - (sum_samples**2)/n_samples)/(n_samples-1)
# returns the average of the samples having the number of samples and sum of the samples
def avg(n_samples, sum_samples):
    return sum_samples/n_samples

# returns the confidence interval with a given confidence
def conf_int(sum_samples, sum_samples_squared, n_samples, confidence):

    # std = sqrt(var)
    std = (var(n_samples, sum_samples, sum_samples_squared))**(1/2)
    mean = avg(n_samples, sum_samples)

    degree_of_freedom = n_samples -1
    t_value = stats.t.ppf(1 - ((1 - confidence) / 2), degree_of_freedom)

    lower_conf = mean - t_value*std/(n_samples**(1/2))
    upper_conf = mean + t_value*std/(n_samples**(1/2))

    return lower_conf, upper_conf

def poly_order_3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c*x + d
def create_poly_order_3_f(a, b, c, d):
    def poly_func(x):
        return a * x**3 + b * x**2 + c*x + d
    return poly_func
def poly_order_1(x, a, b):
    return a * x + b
def create_poly_order_1_f(a, b):
    def poly_func(x):
        return a * x + b
    return poly_func


class Student:
    # class for the simulation of a student with certain characteristics in a
    # university with certain characteristics
    def __init__(self, n_subjects, sessions_per_year,
                 seed, max_exams_per_session,
                 average_num_exams_per_session, min_accuracy_conf_interval,
                 confidence_level, minimum_acceptable_grade, max_years_to_graduate,
                 max_subjects_per_year, get_avg_iq_of_year, year, iq_to_avg_SAT, prints):

        # number of subjects in the degree
        self.n_subjects = n_subjects
        # how many sujects student still has to complete
        self.n_subjects_left = n_subjects
        # number of exam sessions per year
        self.sessions_per_year = sessions_per_year
        # set the seed of the randomness
        self.seed = seed
        np.random.seed(self.seed)

        # maximum number of exams a student can take per session
        # (parameter n of the binomial distribution of how many
        # exams a student will take)
        self.max_exams_per_session = max_exams_per_session

        # for the binomial distribution talked above,
        # p can be calculated by the ration between the avg number of exams the student takes
        # per session and n (E[x]/n = p, for binomial)
        self.probability_of_taking_exam = average_num_exams_per_session/max_exams_per_session

        # minimum accuracy needed to accept an average measure
        self.min_accuracy_conf_interval = min_accuracy_conf_interval
        # confidence level used to compute the accuracy
        self.confidence_level = confidence_level

        # minimum grade a student will accept
        # (minimum is 18, as it is the minimum pass grade)
        self.minimum_acceptable_grade = minimum_acceptable_grade
        if self.minimum_acceptable_grade<18:
            self.minimum_acceptable_grade = 18

        # maximum number of years a student can take to graduate
        # before getting expelled
        self.max_years_to_graduate = max_years_to_graduate

        # store which session the student is at
        self.session_number = 1

        # data storing object
        self.data = Data()

        # maximum subjects a student can take per year
        self.max_subjects_per_year = max_subjects_per_year
        # how many subjects the student has passed already on the year
        self.subjects_taken_on_year = 0
        #function to calculate the average IQ for general population of a given year
        self.get_avg_iq_of_year = get_avg_iq_of_year
        # year
        self.year = year
        # average engineer IQ for the year
        self.avg_iq = self.get_avg_iq_of_year(self.year) +26
        # to store the student's being simulated IQ
        self.student_iq = 0
        # function to compute the average grade of a student from his IQ
        self.iq_to_avg_grade = iq_to_avg_grade
        # maximum average grade, not the maximum grade somebody can take in an exam
        self.max_grade = 30
        # minimum average grade and minimum grade possible
        self.min_grade = 0
        # to score the average grade of a student given his IQ
        self.avg_student_grade = 0

        # to print or not to print inputs and outputs
        self.prints = prints

        # prints all the inputs
        if self.prints:
            print("#######################################")
            print("INPUT PARAMETERS OF THE SIMULATION:\n")
            print(f"number of subjects: {self.n_subjects}")
            print(f"simulating year {self.year}")
            print(f"Average IQ for engineering graduates in the year: {self.avg_iq}")
            print(f"max and min grade, respectively: {self.max_grade, self.min_grade}")
            print(f"number of subjects: {self.n_subjects}")
            print(f"sessions per year: {self.sessions_per_year}")
            print(f"seed: {self.seed}")
            print(f"maximum number of exams per session: {self.max_exams_per_session}")
            print(f"average_num_exams_per_session: {average_num_exams_per_session}")
            print(f"\nminimum accuracy acceptable (for the confidence interval of the averages)\n\t to stop the simulation: {self.min_accuracy_conf_interval}")

            print(f"\t with confidence level used for the confidence intervals: {self.confidence_level}")
            print(f"\nminimum acceptable grade: {self.minimum_acceptable_grade}")
            print(f"maximum number of years a student can take to graduate: {self.max_years_to_graduate}\n\n")

    def get_student_iq(self):
        # returns the student's being simulated IQ from a normal distribution with std = 15
        # and mean calculated from "get_avg_iq_of_year() + 26"
        return np.random.normal(self.avg_iq, 15)


    def get_num_exams_of_session(self):
        """
        Returns the number of exams the student will take in a session
        and it has to be a number between 0 and
        min(number of subjects remaining to graduation, number of subjects took in the year that were not passed)
        """
        exams_to_be_taken = np.random.binomial(self.max_exams_per_session, self.probability_of_taking_exam)
        return min((self.n_subjects_left, exams_to_be_taken, self.max_subjects_per_year-self.subjects_taken_on_year))

    def grade_of_taken_exam(self):
        """
        Returns the grade of the student
        As explained in the report, the grade in an exam is binomially distributed
            with maximum possible grade being 31, but max value for p is 30/31
            (if p=1 was possible, obviously, the student could not get any grade without being the maximum one)
            and n is 31, which we assume is the number of exercises in all exams
        """
        p =  self.avg_student_grade/ (self.max_grade+1)
        return np.random.binomial((self.max_grade+1), p)

    def confidence_intervals_are_good(self):
        """
        Method used to check if all averages have enough acccuracy
        If enough accuracy is achieved, return True and stop the simulation
        """
        # is less than 2 students, cannot compute variance and conf intervals
        if self.data.n_graduated_student < 2:
            return False
        # compute accuracy for average final grade
        (lower, upper) = conf_int(self.data.sum_final_grades, self.data.sum_final_grades_squared, self.data.n_graduated_student, self.confidence_level)
        rel_error = abs((upper-lower)/(avg(self.data.n_graduated_student, self.data.sum_final_grades)*2))
        accuracy1 = (1- rel_error)

        # compute accuracy for average graduation time
        (lower, upper) = conf_int(self.data.sum_graduation_times, self.data.sum_graduation_times_squared, self.data.n_graduated_student, self.confidence_level)
        rel_error = abs((upper-lower)/(avg(self.data.n_graduated_student, self.data.sum_graduation_times)*2))
        accuracy2 = (1- rel_error)

        # compute accuracy for average sessions taken to graduate
        (lower, upper) = conf_int(self.data.sum_sessions_taken, self.data.sum_sessions_taken_squared, self.data.n_graduated_student, self.confidence_level)
        rel_error = abs((upper-lower)/(avg(self.data.n_graduated_student, self.data.sum_sessions_taken)*2))
        accuracy3 = (1- rel_error)

        # in case we get two (or more) identical samples at the start, the accuracy
        # will be 100%, and to prevent that, we don't accept accuracies of 100%
        # (i need to accept accuracy1 of 100% in case a student only accepts 30s)
        if accuracy2==1 or accuracy3==1:
            return False

        # return True if all accuracies are above the minimum accuracy accepted, else, return False
        return (
            (accuracy1>self.min_accuracy_conf_interval) & 
            (accuracy2>self.min_accuracy_conf_interval) & 
            (accuracy3>self.min_accuracy_conf_interval)
        )

    def get_metrics(self):
        """
        This method keeps simulating the same student/university characteristics
        and only returns the output metrics once the accuracies are satisfied
        """

        # keep going until accuracies satisfied
        while not self.confidence_intervals_are_good():

            # will print in stdout the number of students that were simulated (live)
            print(f'\rSTUDENTS SIMULATED: {self.data.n_graduated_student + self.data.non_grad_student}, year: {self.year}', end='', flush=True)
            # get the student's IQ
            self.student_iq = self.get_student_iq()
            # calculate his average grade, clipping to be between [0, 30]
            self.avg_student_grade = max(min(self.iq_to_avg_grade(self.student_iq), self.max_grade), 0)
            # sum of the grades of the student
            sum_grades = 0
            # restart session number
            self.session_number = 1
            # restart the number of subjects left
            self.n_subjects_left = self.n_subjects

            # while the student doens't complete all the subjects and the student doesn't get expelled
            while (self.n_subjects_left != 0) and (self.session_number/6 <= self.max_years_to_graduate):

                # restart the subjects in a year count (once a year)
                if self.session_number%6==1:
                    self.subjects_taken_on_year = 0

                #   get the number of exams the student will try
                n_exams_taken = self.get_num_exams_of_session()

                # get the grade of each exam,
                # and if the student passes, reduce number of subjects left
                # sum the grades with the other grades and increase the number
                # of subjects a student has passed in the year
                for exam in range(n_exams_taken):
                    grade = self.grade_of_taken_exam()

                    if not (grade<self.minimum_acceptable_grade):
                        self.n_subjects_left -= 1
                        sum_grades += grade
                        self.subjects_taken_on_year +=1

                # go to the next session
                self.session_number += 1

            # need to reduce by 1 the session at the end, since we add +1 to
            # the session number before exiting the while loop
            self.session_number -= 1

            # if student finished all the subjects, store the metrics explained in the beginning of the script
            if self.n_subjects_left == 0:

                self.data.sum_final_grades += round(sum_grades/self.n_subjects)

                self.data.sum_final_grades_squared += (round(sum_grades/self.n_subjects))**2
                self.data.sum_graduation_times += self.session_number/6
                self.data.sum_graduation_times_squared += (self.session_number/6)**2
                self.data.final_grade_histogram[round(sum_grades/self.n_subjects)] += 1
                self.data.sum_sessions_taken += self.session_number
                self.data.sum_sessions_taken_squared += (self.session_number)**2
                self.data.graduation_time_histogram[self.session_number/6] += 1
                self.data.grade_by_graduation_time_counts[self.session_number/6][1] +=1
                self.data.grade_by_graduation_time_counts[self.session_number/6][0] += round(sum_grades/self.n_subjects)
                self.data.n_graduated_student += 1

            # if the student did not finish his subjects, means he got expelled
            # so sum to the amount of non graduated students
            else:
                self.data.non_grad_student += 1

        # print/plot all the outputs, as asked in prof. Giaccone's padlet
        if self.prints:

            print("\n\n#########################################################")
            print("SIMULATION RESULTS \n\n")
            print("(the following apply only to student that graduated)")
            print(f"Average student grade: {avg(self.data.n_graduated_student, self.data.sum_final_grades)}")
            print(f"Average student graduation time: {avg(self.data.n_graduated_student, self.data.sum_graduation_times)}")
            print(f"Average number of sessions taken to graduate: {avg(self.data.n_graduated_student, self.data.sum_sessions_taken)}")
            print(f"Percentage of students who could not graduate: {100*self.data.non_grad_student/(self.data.non_grad_student+ self.data.n_graduated_student)}")
            print()
            plt.figure(figsize=(15, 5))

            # Plot 1: Final Grade Histogram
            plt.subplot(1, 3, 1)
            plt.bar(self.data.final_grade_histogram.keys(), self.data.final_grade_histogram.values())
            plt.title("Final Grade Histogram")
            plt.xlabel("Grade")
            plt.ylabel("Number of Students")
            plt.grid()

            # Plot 2: Graduation Time Histogram
            plt.subplot(1, 3, 2)
            plt.bar(self.data.graduation_time_histogram.keys(), self.data.graduation_time_histogram.values(), width = 0.13)
            plt.title("Graduation Time Histogram")
            plt.xlabel("Years to Graduate")
            plt.ylabel("Number of Students")
            plt.grid()

            # Plot 3: Grade by Graduation Time
            plt.subplot(1, 3, 3)
            grade_by_graduation_time_counts = list(self.data.grade_by_graduation_time_counts.items())
            grade_by_graduation_time_counts.sort()
            times = [pair[0] for pair in grade_by_graduation_time_counts]
            ratios = [pair[1][0] / pair[1][1] for pair in grade_by_graduation_time_counts]
            plt.plot(times, ratios, marker='o')
            plt.title("Average grade by Graduation Time ")
            plt.xlabel("Years to Graduate")
            plt.ylabel("Average Grade")
            plt.grid()

            plt.tight_layout()
            plt.savefig("1990.png")
            plt.show()
        return self.data

#######################################################
# COMPUTING THE RELATION BETWEEN YEAR AND AVERAGE IQ
# by using the data from doi: 10.1177/1745691615577701
#######################################################

# data from the paper, where the values are how much the IQ from the general
# increased PER YEAR
time_periods = {
    (1924, 1935): 0.93,
    (1936, 1938): 0.58,
    (1939, 1952): 0.20,
    (1953, 1985): 0.43,
    (1986, 2013): 0.22
}

# initialize the list to hold IQ scores, starting with the year 1924 having an IQ score of 100
iq_scores = [100]

# starting year
current_year = 1924

# calculate the IQ scores for each year based on the increases provided
for period, increase in time_periods.items():
    start, end = period
    for year in range(start, end + 1):
        if year == current_year:
            # skip the first year of each period since it's already added
            continue
        # calculate the new IQ score and append it to the list
        iq_scores.append(iq_scores[-1] + increase)
        current_year = year

# get the list of years from 1924 to the last period's end year
years = list(range(1924, current_year + 1))

iq_scores = np.array(iq_scores)

# just to rename
x = years
y = iq_scores

# use curve_fit to fit a third-order polynomial on the data
params, covariance = curve_fit(poly_order_3, x, y)

# get the parameters
a, b, c, d = params

# create a function that calculates the average IQ of a generation
# using 1924's metrics
get_avg_iq_of_year = create_poly_order_3_f(a, b, c, d)

# get the average IQ of the generation population in 2023
avg_iq_2023 = get_avg_iq_of_year(2023)

# normalize the parameters in such a way that the general population from 2023
# has an IQ of 100
a, b, c, d = [param*100/avg_iq_2023 for param in params]  # params obtained from curve_fit or elsewhere

# create a function that calculates the average IQ of a generation
# using 2023's metrics
get_avg_iq_of_year = create_poly_order_3_f(a, b, c, d)

##################################################################
# COMPUTING THE RELATION BETWEEN IQ AND AVERAGE GRADE
# by using the data from doi: 10.1111/j.0956-7976.2004.00687.x
##################################################################

# the range of the data of SAT scores in the paper
y = np.linspace(400, 1337, 1000)

# the function used to predict IQ given an SAT score in the paper
x = (0.126*y) + (-4.71e-5 * (y**2)) + 40.063

# fit a first-order polynomial on the inverted equation in the range
# where the function is bijective
params, covariance = curve_fit(poly_order_1, x, y)

a, b = params  # params obtained from curve_fit

# normalize the parameters to give grades in the range [0, 30]
# (maximum and minimum values for SAT are, respectively, 1600 and 400)
a = a * 30/(1600-400)
b = (b-400)*30/(1600-400)

# create the function that calculates the average grade from the IQ of a person
iq_to_avg_grade = create_poly_order_1_f(a, b)

#########################################
# SETTING THE INPUT PARAMETERS
#########################################

n_subjects = 16
sessions_per_year = 6
seed = 42
max_exams_per_session = 4
average_num_exams_per_session = 2
min_accuracy_conf_interval = 0.99
confidence_level = 0.99
minimum_acceptable_grade = 18
max_years_to_graduate = 4
max_subjects_per_year = 8

year = 1924

# if true, print all the inputs and output metrics
# this is useful for when doing loops on different parameters,
# so that not a lot of things is printed
prints = True

    # initialize the simulator with the parameters
stud = Student(n_subjects, sessions_per_year,
                seed, max_exams_per_session,
                average_num_exams_per_session, min_accuracy_conf_interval,
                confidence_level, minimum_acceptable_grade, max_years_to_graduate, max_subjects_per_year, get_avg_iq_of_year, year, iq_to_avg_grade, prints)

# simulate and get metrics (they are printed anyway)
data = stud.get_metrics()