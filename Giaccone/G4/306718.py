import numpy as np
import matplotlib.pyplot as plt
from math import comb
from scipy import stats

# grades distribution
MIN_GRADE, MAX_GRADE = 18,30
GRADES_DIST = { 18:87,  19:62,  20:74,  21:55,  22:99, 23:94,  24:117, 
                25:117, 26:136, 27:160, 28:215, 29:160, 30:473 }
# dict with cumulative probability for each possible grade
GRADES_CUMULAT_PROB = dict()
n_samples = sum(GRADES_DIST.values())
cumulat_prob = 0
for grd in range(MIN_GRADE,MAX_GRADE+1):
    cumulat_prob += GRADES_DIST[grd] / n_samples
    GRADES_CUMULAT_PROB[grd] = cumulat_prob
# generate a grade, based on the cumulative distribution
def grade_generator():
    u = np.random.uniform(0,1)
    for grd in range(MIN_GRADE,MAX_GRADE+1):
        if u <= GRADES_CUMULAT_PROB[grd]:
            return grd

def exam_success(approval_prob):
    # bernoulli experiment with parameter p=approval_prob
    return np.random.uniform(0,1) < approval_prob

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
    p = 0.5
    n = round( avg_exams_per_session / p )
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

def simulation(total_courses, avg_exams_per_session, 
               approval_prob, sessions_per_year):
    # perform a simulation of the graduation of a single student

    grades=np.zeros(total_courses, dtype=np.int32)
    exams_left = total_courses
    n_sessions = 0

    # loop until there are no pending exams to pass
    while exams_left > 0:
        # exam session - stochastic elements
        # number of exams taken during the session as a binomial random variable
        exams_taken = exams_per_session_gen(exams_left, avg_exams_per_session)
        for exam in range(exams_taken):
            # pass or not a exam according to a bernoulli random variable
            if exam_success(approval_prob):
                # student grade according to the given distribution
                grades[total_courses-exams_left] = grade_generator()
                exams_left -= 1
        n_sessions += 1
    
    final_grade = grades.mean()
    graduation_time = n_sessions / sessions_per_year
    return final_grade, graduation_time

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

def n_simulations(total_courses, avg_exams_per_session, 
                  approval_prob, sessions_per_year,
                  conf_level, min_accuracy, verbose=False):
    # perform simulations until the minimum accuracy is met for both metrics 
    acc_grade, acc_time = 0, 0

    # get the values of the first simulation
    final_grade, graduation_time = simulation(total_courses, 
                                              avg_exams_per_session, 
                                              approval_prob, 
                                              sessions_per_year)
    n_simulations = 1
    grades = [ final_grade ]
    graduation_times = [ graduation_time ]

    if verbose:
        print('INPUT PARAMETERS:')
        print(f'Probability of success at each exam: {approval_prob}')
        print(f'Average exams taken at each session: {avg_exams_per_session}')
        print(f'Total number of courses to graduate: {total_courses}')
        print(f'Exam sessions per year: {sessions_per_year}')
        print(f'Minimum accuracy of output measures: {min_accuracy}\n')

    while min(acc_grade, acc_time) < min_accuracy:
        final_grade, graduation_time = simulation(total_courses, 
                                                  avg_exams_per_session, 
                                                  approval_prob, 
                                                  sessions_per_year)
        n_simulations += 1
        grades.append(final_grade)
        graduation_times.append(graduation_time)

        # confidence interval and accuracy computation
        avg_grade, delta_grade = confidence_interval(grades, conf_level)
        avg_time, delta_time = confidence_interval(graduation_times, conf_level)
        new_acc_grade = accuracy(avg_grade, delta_grade)
        new_acc_time = accuracy(avg_time, delta_time)

        # record number of simulations needed for each measure
        if new_acc_grade>=min_accuracy and acc_grade<min_accuracy:
            grade_nsims = n_simulations
        if new_acc_time>=min_accuracy and acc_time<min_accuracy:
            time_nsims = n_simulations
        acc_grade, acc_time = new_acc_grade, new_acc_time

    if verbose:
        print('OUTPUT MEASURES')
        print(f'Number of simulations until minimum accuracy:\n\t- grade: {grade_nsims}\n\t- graduation time: {time_nsims}')
        print(f'Average graduation grade: {avg_grade:.2f} \u00b1 {delta_grade:.2f}')
        print(f'Average graduation time: {avg_time:.2f} \u00b1 {delta_time:.2f} years')

    grades_result = avg_grade, delta_grade
    graduation_time_results = avg_time, delta_time
    return grades_result, graduation_time_results

def plot_parameter(parameter_values, graduation_time_values, parameter_name, graduation_time_deltas=None):
    # plot parameter_values and the respective graduation time
    # if graduation_time_deltas is given, plot also the error bar
    
    plt.plot(parameter_values, graduation_time_values, '.r', label='Average graduation time')
    if graduation_time_deltas is not None:
        plt.errorbar(
            x=parameter_values,
            y=graduation_time_values,
            yerr=graduation_time_deltas,
            color='blue',
            linestyle='none',
            label='Graduation time error'
        )
    
    plt.xlabel(parameter_name)
    plt.ylabel('Years')
    plt.ylim([0,graduation_time_values.max()+1])
    plt.yticks(np.arange(0,graduation_time_values.max()+1,.5))
    plt.legend(fancybox=True, framealpha=1)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    seed = 40028922
    np.random.seed(seed)
    CONFIDENCE_LEVEL = 0.98

    print('SELECT AN EXPERIMENT:')
    print('1 - validation - results for a single set of input parameters')
    print('2 - graduation time vs. probability of exam success')
    print('3 - graduation time vs. average exams taken by exam session')
    experiment = input()
    print('\n#################')
    
    if experiment == '1':
        print("EXPERIMENT 1: validation - single results")
        # Input parameters
        min_accuracy = 0.98
        avg_exams_per_session = 3.0
        total_courses = 18
        approval_prob = 0.55
        sessions_per_year = 3

        n_simulations(
            min_accuracy=min_accuracy, 
            conf_level=CONFIDENCE_LEVEL, 
            total_courses=total_courses, 
            avg_exams_per_session=avg_exams_per_session, 
            approval_prob=approval_prob, 
            sessions_per_year=sessions_per_year,
            verbose=True
        )

    elif experiment == '2':
        print('EXPERIMENT 2: graduation time vs. probability of exam success')
        # Input parameters
        min_accuracy = 0.98
        avg_exams_per_session = 3.5
        total_courses = 16
        sessions_per_year = 3
        print('INPUT PARAMETERS:')
        print(f'Average exams taken at each session: {avg_exams_per_session}')
        print(f'Total number of courses to graduate: {total_courses}')
        print(f'Exam sessions per year: {sessions_per_year}')
        print(f'Minimum accuracy of output measures: {min_accuracy}\n')

        # considered values
        approval_probs = np.arange(0.30,.81,0.05)
        
        n_values = len(approval_probs)
        gradtime_avg = np.zeros( n_values )
        gradtime_delta = np.zeros( n_values )
        
        for i,p in enumerate(approval_probs):
            # simulate each value until minimum accuracy is met
            grade, graduation_time = n_simulations(min_accuracy=min_accuracy, 
                                            conf_level=CONFIDENCE_LEVEL, 
                                            total_courses=total_courses, 
                                            avg_exams_per_session=avg_exams_per_session, 
                                            approval_prob=p, 
                                            sessions_per_year=sessions_per_year,
                                            verbose=False)
            gradtime_avg[i] = graduation_time[0]
            gradtime_delta[i] = graduation_time[1]
        plot_parameter(
            approval_probs, 
            gradtime_avg, 
            'Exam success probability',
            graduation_time_deltas=gradtime_delta
        )

    elif experiment == '3':
        print('EXPERIMENT 3: graduation time vs. average exams taken by exam session')
        # Input parameters
        min_accuracy = 0.98
        total_courses = 14
        approval_prob = 0.55
        sessions_per_year = 3
        print('INPUT PARAMETERS:')
        print(f'Total number of courses to graduate: {total_courses}')
        print(f'Probability of success at each exam: {approval_prob}')
        print(f'Exam sessions per year: {sessions_per_year}')
        print(f'Minimum accuracy of output measures: {min_accuracy}\n')

        # considered values
        avg_exams_per_session_list = np.arange(1.5,4.6,0.2)
        
        n_values = len(avg_exams_per_session_list)
        gradtime_avg = np.zeros( n_values )
        gradtime_delta = np.zeros( n_values )

        for i,avg_exams_session in enumerate(avg_exams_per_session_list):
            # simulate each value until minimum accuracy is met
            grade, graduation_time = n_simulations(min_accuracy=min_accuracy, 
                                            conf_level=CONFIDENCE_LEVEL, 
                                            total_courses=total_courses, 
                                            avg_exams_per_session=avg_exams_session, 
                                            approval_prob=approval_prob, 
                                            sessions_per_year=sessions_per_year,
                                            verbose=False)
            gradtime_avg[i] = graduation_time[0]
            gradtime_delta[i] = graduation_time[1]
        plot_parameter(
            avg_exams_per_session_list, 
            gradtime_avg, 
            'Average exams taken by session',
            graduation_time_deltas=gradtime_delta
        )