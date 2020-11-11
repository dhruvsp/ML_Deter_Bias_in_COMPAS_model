from utils import *
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: # Accuracy
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
def enforce_demographic_parity(categorical_results, epsilon):
    accuracy = 0
    data_copy = {}
    demographic_parity_data = {}
    thresholds = {'African-American': 0, 'Caucasian': 0, 'Hispanic': 0, 'Other': 0}
    thr_dummy = []
    afr_am_npp = []
    caucasian_npp = []
    hispanic_npp = [] 
    others_npp = []
    thr_test = []
    
    threshold_list = [float(j) / 100 for j in range(0, 100, 1)]
    for i in threshold_list:
        thr_dummy.append(i)
        data_copy['African-American'] = apply_threshold(categorical_results['African-American'], i)
        data_copy['Caucasian'] = apply_threshold(categorical_results['Caucasian'], i)
        data_copy['Hispanic'] = apply_threshold(categorical_results['Hispanic'], i)
        data_copy['Other'] = apply_threshold(categorical_results['Other'], i)
        afr_am_npp.append(get_num_predicted_positives(data_copy['African-American'])/len(data_copy['African-American']))
        caucasian_npp.append(get_num_predicted_positives(data_copy['Caucasian'])/len(data_copy['Caucasian']))
        hispanic_npp.append(get_num_predicted_positives(data_copy['Hispanic'])/len(data_copy['Hispanic']))
        others_npp.append(get_num_predicted_positives(data_copy['Other'])/len(data_copy['Other']))
    for afr_am_prob in afr_am_npp:
        for cauc_prob in caucasian_npp:
            if prob_comparator(afr_am_prob, cauc_prob, epsilon) == False:
                continue
            for his_prob in hispanic_npp:
                if  prob_comparator(cauc_prob, his_prob, epsilon) == False:
                    continue
                if  prob_comparator(afr_am_prob, his_prob, epsilon) == False:
                    continue    
                for oth_prob in others_npp:
                    if prob_comparator(afr_am_prob, oth_prob, epsilon) == False:
                        continue
                    if prob_comparator(cauc_prob, oth_prob, epsilon) == False:
                        continue
                    if prob_comparator(his_prob, oth_prob, epsilon) == False:
                        continue
                    else:
                        for i, j in enumerate(afr_am_npp):
                            if j == afr_am_prob:
                                index_val = i
                        thr_dum1 = thr_dummy[index_val]   
                        for i, j in enumerate(caucasian_npp):
                            if j == cauc_prob:
                                index_valc = i
                        thr_dum2 = thr_dummy[index_valc]
                        for i, j in enumerate(hispanic_npp):
                            if j == his_prob:
                                index_valh = i
                        thr_dum3 = thr_dummy[index_valh]
                        for i, j in enumerate(others_npp):
                            if j == oth_prob:
                                index_valo = i
                        thr_dum4 = thr_dummy[index_valo]  
                        thr_positive = [thr_dum1, thr_dum2 , thr_dum3 , thr_dum4]
                        if thr_positive not in thr_test:
                            thr_test.append(thr_positive)
    for thresh in thr_test:
        data_copy['African-American'] = apply_threshold(categorical_results['African-American'], thresh[0])
        data_copy['Caucasian'] = apply_threshold(categorical_results['Caucasian'], thresh[1])
        data_copy['Hispanic'] = apply_threshold(categorical_results['Hispanic'], thresh[2])
        data_copy['Other'] = apply_threshold(categorical_results['Other'], thresh[3])
        total_accuracy = get_total_accuracy(data_copy)
        if total_accuracy > accuracy:
            accuracy = total_accuracy
            thresholds = {'African-American': thresh[0], 'Caucasian': thresh[1], 'Hispanic': thresh[2], 'Other': thresh[3]}                 
    for key in categorical_results.keys():
        threshold = thresholds[key]
        demographic_parity_data[key] = apply_threshold(categorical_results[key], threshold)
    return demographic_parity_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):
    accuracy = 0
    data_copy = {}
    equal_opportunity_data = {}
    thresholds = {'African-American': 0, 'Caucasian': 0, 'Hispanic': 0, 'Other': 0}
    thr_dummy = []
    afr_am_tpr = []
    caucasian_tpr = []
    hispanic_tpr = []
    other_tpr = []
    thr_test = []
    
    threshold_list = [float(j) / 100 for j in range(0, 100, 1)]
    for i in threshold_list:
        thr_dummy.append(i)
        data_copy['African-American'] = apply_threshold(categorical_results['African-American'], i)
        data_copy['Caucasian'] = apply_threshold(categorical_results['Caucasian'], i)
        data_copy['Hispanic'] = apply_threshold(categorical_results['Hispanic'], i)
        data_copy['Other'] = apply_threshold(categorical_results['Other'], i)
        afr_am_tpr.append(get_true_positive_rate(data_copy['African-American']))
        caucasian_tpr.append(get_true_positive_rate(data_copy['Caucasian']))
        hispanic_tpr.append(get_true_positive_rate(data_copy['Hispanic']))
        other_tpr.append(get_true_positive_rate(data_copy['Other']))
    for afr_am_prob in afr_am_tpr:
        for cauc_prob in caucasian_tpr:
            if prob_comparator(afr_am_prob, cauc_prob, epsilon) == False:
                continue
                for his_prob in hispanic_tpr:
                    if  prob_comparator(cauc_prob, his_prob, epsilon) == False:
                        continue
                    if  prob_comparator(afr_am_prob, his_prob, epsilon) == False:
                        continue    
                    for oth_prob in others_npp:
                        if prob_comparator(afr_am_prob, oth_prob, epsilon) == False:
                            continue
                        if prob_comparator(cauc_prob, oth_prob, epsilon) == False:
                            continue
                        if prob_comparator(his_prob, oth_prob, epsilon) == False:
                            continue
                        else:
                            for i, j in enumerate(afr_am_tpr):
                                if j == afr_am_prob:
                                    index_val = i
                            thr_dum1 = thr_dummy[index_val]   
                            for i, j in enumerate(caucasian_tpr):
                                if j == cauc_prob:
                                    index_valc = i
                            thr_dum2 = thr_dummy[index_valc]
                            for i, j in enumerate(hispanic_tpr):
                                if j == his_prob:
                                    index_valh = i
                            thr_dum3 = thr_dummy[index_valh]
                            for i, j in enumerate(others_tpr):
                                if j == oth_prob:
                                    index_valo = i
                            thr_dum4 = thr_dummy[index_valo]  
                            thr_positive = [thr_dum1, thr_dum2 , thr_dum3 , thr_dum4]
                            if thr_positive not in thr_test:
                                thr_test.append(thr_positive)
                                
    for thresh in thr_test:
        data_copy['African-American'] = apply_threshold(categorical_results['African-American'], thresh[0])
        data_copy['Caucasian'] = apply_threshold(categorical_results['Caucasian'], thresh[1])
        data_copy['Hispanic'] = apply_threshold(categorical_results['Hispanic'], thresh[2])
        data_copy['Other'] = apply_threshold(categorical_results['Other'], thresh[3])
        total_accuracy = get_total_accuracy(data_copy)
        if total_accuracy > accuracy:
            accuracy = total_accuracy
            thresholds = {'African-American': thresh[0], 'Caucasian': thresh[1], 'Hispanic': thresh[2], 'Other': thresh[3]}                
    for key in categorical_results.keys():
        threshold = thresholds[key]
        equal_opportunity_data[key] = apply_threshold(categorical_results[key], threshold)
    return equal_opportunity_data, thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):

    accuracy = 0
    data_copy = {}
    maximum_profit_data = {}
    thresholds = {'African-American': 0, 'Caucasian': 0, 'Hispanic': 0, 'Other': 0}
    afr_am_max_prof = 0
    caucasian_max_prof = 0
    hispanic_max_prof = 0
    others_max_prof = 0 
    
    threshold_list = [float(j) / 100 for j in range(0, 100, 1)]
    for i in threshold_list:
        data_copy['African-American'] = apply_threshold(categorical_results['African-American'], i)        
        data_copy['Caucasian'] = apply_threshold(categorical_results['Caucasian'], i)      
        data_copy['Hispanic'] = apply_threshold(categorical_results['Hispanic'], i)     
        data_copy['Other'] = apply_threshold(categorical_results['Other'], i)        
        af_acc = get_num_correct(data_copy['African-American']) / len(data_copy['African-American'])
        cauc_acc = get_num_correct(data_copy['Caucasian']) / len(data_copy['Caucasian'])
        his_acc = get_num_correct(data_copy['Hispanic']) / len(data_copy['Hispanic'])  
        oth_acc = get_num_correct(data_copy['Other']) / len(data_copy['Other'])
        if af_acc > afr_am_max_prof:
            afr_am_max_prof = af_acc
            thresholds['African-American'] = i
        if cauc_acc > caucasian_max_prof:
            caucasian_max_prof = cauc_acc
            thresholds['Caucasian'] = i       
        if his_acc > hispanic_max_prof:
            hispanic_max_prof = his_acc
            thresholds['Hispanic'] = i       
        if oth_acc > others_max_prof:
            others_max_prof = oth_acc
            thresholds['Other'] = i
    for key in categorical_results.keys():
        threshold = thresholds[key]
        maximum_profit_data[key] = apply_threshold(categorical_results[key], threshold)
    return maximum_profit_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    data_copy, predictive_parity_data = {}, {}
    thresholds = {'African-American': 0, 'Caucasian': 0, 'Hispanic': 0, 'Other': 0}
    accuracy = 0
    thr_dummy = []
    afr_am_ppv = []
    caucasian_ppv = []
    hispanic_ppv = []
    others_ppv = []
    thr_test = []
    threshold_list = [j / 100 for j in range(0, 100, 1)]
    for i in threshold_list:
        thr_dummy.append(i)
        
        data_copy['African-American'] = apply_threshold(categorical_results['African-American'], i)
        afr_am_ppv.append(get_positive_predictive_value(data_copy['African-American']))

        data_copy['Caucasian'] = apply_threshold(categorical_results['Caucasian'], i)
        caucasian_ppv.append(get_positive_predictive_value(data_copy['Caucasian']))

        data_copy['Hispanic'] = apply_threshold(categorical_results['Hispanic'], i)
        hispanic_ppv.append(get_positive_predictive_value(data_copy['Hispanic']))

        data_copy['Other'] = apply_threshold(categorical_results['Other'], i)
        others_ppv.append(get_positive_predictive_value(data_copy['Other']))
    for afr_am_prob in afr_am_ppv:
        for cauc_prob in caucasian_ppv:
            if prob_comparator(afr_am_prob, cauc_prob, epsilon) == False:
                continue
            for his_prob in hispanic_ppv:
                if  prob_comparator(cauc_prob, his_prob, epsilon) == False:
                    continue
                if  prob_comparator(afr_am_prob, his_prob, epsilon) == False:
                    continue    
                for oth_prob in others_ppv:
                    if prob_comparator(afr_am_prob, oth_prob, epsilon) == False:
                        continue
                    if prob_comparator(cauc_prob, oth_prob, epsilon) == False:
                        continue
                    if prob_comparator(his_prob, oth_prob, epsilon) == False:
                        continue
                    else:
                        for i, j in enumerate(afr_am_ppv):
                            if j == afr_am_prob:
                                index_val = i
                        thr_dum1 = thr_dummy[index_val]   
                        for i, j in enumerate(caucasian_ppv):
                            if j == cauc_prob:
                                index_valc = i
                        thr_dum2 = thr_dummy[index_valc]
                        for i, j in enumerate(hispanic_ppv):
                            if j == his_prob:
                                index_valh = i
                        thr_dum3 = thr_dummy[index_valh]
                        for i, j in enumerate(others_ppv):
                            if j == oth_prob:
                                index_valo = i
                        thr_dum4 = thr_dummy[index_valo]  
                        thr_positive = [thr_dum1, thr_dum2 , thr_dum3 , thr_dum4]
                        if thr_positive not in thr_test:
                            thr_test.append(thr_positive)
    for thresh in thr_test:
        data_copy['African-American'] = apply_threshold(categorical_results['African-American'], thresh[0])
        data_copy['Caucasian'] = apply_threshold(categorical_results['Caucasian'], thresh[1])
        data_copy['Hispanic'] = apply_threshold(categorical_results['Hispanic'], thresh[2])
        data_copy['Other'] = apply_threshold(categorical_results['Other'], thresh[3])
        total_accuracy = get_total_accuracy(data_copy)
        if total_accuracy > accuracy:
            accuracy = total_accuracy
            thresholds = {'African-American': thresh[0], 'Caucasian': thresh[1], 'Hispanic': thresh[2], 'Other': thresh[3]}                
    for key in categorical_results.keys():
        threshold = thresholds[key]
        predictive_parity_data[key] = apply_threshold(categorical_results[key], threshold)
    return predictive_parity_data, thresholds

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    data_copy = {}
    accuracy = 0
    threshold_list = [j / 100 for j in range(0, 100, 1)]
    for i in threshold_list:
        for key in categorical_results.keys():
            data_copy[key] = apply_threshold(categorical_results[key], i)
        total_accuracy = get_total_accuracy(data_copy)
        if total_accuracy > accuracy:
            accuracy = total_accuracy
            thresholds = {'African-American': i, 'Caucasian': i, 'Hispanic': i, 'Other': i}

    single_threshold_data = {}
    for key in categorical_results.keys():
        threshold = thresholds[key]
        single_threshold_data[key] = apply_threshold(categorical_results[key], threshold)

    return single_threshold_data, thresholds

def prob_comparator(p1, p2, epsilon):
    return abs(p1 - p2) <= epsilon
