import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sympy import simplify_logic, parse_expr, symbols
import copy

class Guess(object):

    def __init__(self, config):
        self.config = config


    def rank(self,predictions, key, targets, ntraces, interval=10):
        ranktime = np.zeros(int(ntraces / interval))
        pred = np.zeros(256)

        idx = np.random.randint(predictions.shape[0], size=ntraces)

        for i, p in enumerate(idx):
            for k in range(predictions.shape[1]):
                pred[k] += predictions[p, targets[p, k]]

            if i % interval == 0:
                ranked = np.argsort(pred)[::-1]
                ranktime[int(i / interval)] = list(ranked).index(key)

        return ranktime

    def prediction(self, predictions,n_attack,n_traces,interval,key_attack,targets):
        predictions = F.softmax(predictions, dim=1)
        predictions = predictions.cpu().detach().clone().numpy()
        predictions = np.log(predictions)
        # print(predictions)

        ranks_zaid = np.zeros((n_attack, int(n_traces / interval)))
        for i in tqdm(range(n_attack)):
            ranks_zaid[i] = self.rank(predictions, key_attack, targets, n_traces, interval)
        return ranks_zaid

    def guess(self, dataloader, model,device, n_traces = 400, n_attack = 100 ,interval = 1):
        X_attack = torch.Tensor(dataloader.X_profiling).unsqueeze(1).to(device)
        targets = dataloader.Y_profiling
        key_attack = dataloader.real_key
        #tracesAttack_shaped = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
        predictions = model(X_attack)
        ranks_zaid = self.prediction(predictions, n_attack, n_traces, interval, key_attack, targets)
        return ranks_zaid

    def guess_sat(self, dataloader, model,device, sat_exp, W_prunned, unfoldb0, n_traces = 400, n_attack = 100 ,interval = 1): ##The sat_exp should be 'dnf'.
        targets = dataloader.Y_profiling
        key_attack = dataloader.real_key
        print(torch.Tensor(dataloader.X_profiling).shape)
        X_attack = torch.Tensor(dataloader.X_profiling).unsqueeze(1).to(device)
        _ = model(X_attack)
        model_feat = model.feat.clone() #model features obtain after flatten 0-1 from the model before the linear regression
        print(X_attack.shape)
        X_attack = torch.Tensor(dataloader.X_profiling).unsqueeze(1).to(device)
        print(X_attack.shape)
        X_attack = model.preprocessing(X_attack.float()).detach().clone()
        X_attack = X_attack.unsqueeze(-1)
        X_attack = unfoldb0(X_attack) ##[batch,kernel size =9, no of kernel with stripe 4 = 7 (patch)]
        num_features = model.cal_num_features(self.config)
        num_filters =self.config.layer_2[0]
        features = np.zeros((X_attack.shape[0], num_features*num_filters))
        ##Guess attack using Sat_expression.

        for batch in tqdm(range(X_attack.shape[0])):
            cpt = 0
            for filter in range(num_filters):
                exp_DNF = parse_expr(sat_exp[filter])
                for time in range(num_features):
                    patch = X_attack[batch, :, time]
                    evaluate = {}
                    for value_input2iciindex in range(9):
                        evaluate["x_" + str(value_input2iciindex)] = bool(int(patch[value_input2iciindex]))
                    exp_DNFici = simplify_logic(exp_DNF.subs(evaluate), form='dnf')
                    if str(exp_DNFici) == "True":
                        features[batch, cpt] = 1
                    elif exp_DNFici in [1.0]: #if exp_DNFici is a constant and =1 then set it as 1
                        features[batch, cpt] = 1
                    cpt += 1
        print((model_feat.cpu().detach().numpy() == features).all())
        if model.fc1.bias is not None:
            prediction2 = torch.Tensor(np.dot(W_prunned, features.T).T) + model.fc1.bias.cpu().detach().clone().numpy()
        else:
            prediction2 = torch.Tensor(np.dot(W_prunned, features.T).T)
        ranks_zaid = self.prediction(prediction2, n_attack, n_traces, interval, key_attack, targets)

        return ranks_zaid




    def guess_sat_table(self,PATH_expr,name, dataloader, model,device, sat_exp_TT, W_prunned, unfoldb0, n_traces = 400, n_attack = 100 ,interval = 1, features_given = None): ##The sat_exp should be 'dnf'.
        targets = dataloader.Y_profiling
        key_attack = dataloader.real_key
        print(torch.Tensor(dataloader.X_profiling).shape)
        X_attack = torch.Tensor(dataloader.X_profiling).unsqueeze(1).to(device)
        print(X_attack.shape)
        _ = model(X_attack)
        model_feat = model.feat.clone() #model features obtain after flatten 0-1 from the model before the linear regression
        print(model_feat.shape)
        # X_attack = torch.Tensor(dataloader.X_profiling).unsqueeze(1).to(device)
        print(X_attack.shape)
        X_attack = model.preprocessing(X_attack.float()).detach().clone()
        X_attack = X_attack.unsqueeze(-1)
        X_attack = unfoldb0(X_attack) ##[batch,kernel size =9, no of kernel with stripe 4 = 7 (patch)]
        print(X_attack.shape)
        num_features = model.cal_num_features(self.config)
        num_filters =self.config.layer_2[0]
        features = np.zeros((X_attack.shape[0], num_features*num_filters))
        ##Guess attack using Sat_expression.
        if features_given != None:
            features = features_given
        else:
            for batch in tqdm(range(X_attack.shape[0])):
                cpt = 0
                for filter in range(num_filters):
                    # exp_DNF = sat_exp[filter]
                    TruthTable_perfilter = sat_exp_TT[filter]
                    for time in range(num_features):
                        patch = X_attack[batch, :, time]
                        evaluate = 0
                        for value_input2iciindex in range(9):
                            evaluate = (evaluate << 1) | int(patch[value_input2iciindex])
                            #evaluate["x_" + str(value_input2iciindex)] = bool(int(patch[value_input2iciindex]))
                        # exp_DNFici = simplify_logic(exp_DNF.subs(evaluate), form='dnf')
                        exp_DNFici = TruthTable_perfilter[evaluate]
                        if str(exp_DNFici) == "True":
                            features[batch, cpt] = 1
                        elif str(exp_DNFici) != "False":
                            if float(exp_DNFici) in [1.0]:  # if exp_DNFici is a constant and =1 then set it as 1
                                features[batch, cpt] = 1
                        cpt += 1
            torch.save(features, os.path.join(PATH_expr, "features_" + name + '.pt'))

        print((model_feat.cpu().detach().numpy()))
        print(features)
        print((model_feat.cpu().detach().numpy() == features))
        print((model_feat.cpu().detach().numpy() == features).all())
        if model.fc1.bias is not None:
            prediction2 = torch.Tensor(np.dot(W_prunned, features.T).T) + model.fc1.bias.cpu().detach().clone().numpy()
        else:
            prediction2 = torch.Tensor(np.dot(W_prunned, features.T).T)
        ranks_zaid = self.prediction(prediction2, n_attack, n_traces, interval, key_attack, targets)
        return ranks_zaid





    '''
    mode: either 'dnf' or 'cnf'
    '''
    def guess_sat_and_count_activated_mask(self,dataloader, model, device, sat_exp, W_prunned, unfoldb0, mode, n_traces = 400, n_attack = 100 ,interval = 1):
        targets = dataloader.Y_profiling
        key_attack = dataloader.real_key
        X_attack = torch.Tensor(dataloader.X_profiling).unsqueeze(1)
        _ = model(X_attack)
        model_feat = model.feat.clone() #model features obtain after flatten 0-1 from the model before the linear regression
        X_attack = torch.Tensor(dataloader.X_profiling).unsqueeze(1).to(device)
        X_attack = model.preprocessing(X_attack.float()).detach().clone()
        X_attack = unfoldb0(X_attack.unsqueeze(-1)) ##[batch,kernel size =9, no of kernel with stripe 4 = 7 (patch)]
        num_features = model.cal_num_features(self.config)
        num_filters =self.config.layer_2[0]
        features = np.zeros((X_attack.shape[0], num_features*num_filters))
        ##initialized all the mask into a dictionary
        keys = []
        for filter in range(num_filters):
            str_exp_DNF = str(sat_exp[filter])
            if mode == 'dnf':
                mask_per_filter = str_exp_DNF.split("|")
            elif mode == 'cnf':
                mask_per_filter = str_exp_DNF.split("&")
            keys.extend(mask_per_filter)
        keys = np.array(keys)
        keys = np.unique(keys)
        nbre_mask_activated_per_filter = dict.fromkeys(keys, 0)
        #Caulculate the guess.
        for batch in tqdm(range(X_attack.shape[0])):
            cpt = 0
            for filter in range(num_filters):
                exp_DNF = sat_exp[filter]
                str_exp_DNF = str(exp_DNF)
                if mode == 'dnf':
                    mask_per_filter = str_exp_DNF.split("|")
                elif mode == 'cnf':
                    mask_per_filter = str_exp_DNF.split("&")
                for mask in mask_per_filter:
                    mask_logic = simplify_logic(mask, form=mode)
                    for time in range(num_features):
                        patch = X_attack[batch, :, time]
                        evaluate_mask = {}
                        for value_input2iciindex in range(9):
                            evaluate_mask["x_" + str(value_input2iciindex)] = bool(int(patch[value_input2iciindex]))
                        exp_DNF_mask = mask_logic.subs(evaluate_mask)
                        if str(exp_DNF_mask) == "True":  # Then the mask is activated.
                            nbre_mask_activated_per_filter[mask] += 1

                for time in range(num_features):
                    patch = X_attack[batch, :, time]
                    evaluate = {}
                    for value_input2iciindex in range(9):
                        evaluate["x_" + str(value_input2iciindex)] = bool(int(patch[value_input2iciindex]))
                    exp_DNFici = simplify_logic(exp_DNF.subs(evaluate), form='dnf')
                    if str(exp_DNFici) == "True":
                        features[batch, cpt] = 1

                    elif exp_DNFici in [1.0]: #if exp_DNFici is a constant and =1 then set it as 1
                        features[batch, cpt] = 1
                    cpt += 1
        print((model_feat.cpu().detach().numpy() == features).all())
        if model.fc1.bias is not None:
            prediction2 = torch.Tensor(np.dot(W_prunned, features.T).T) + model.fc1.bias.cpu().detach().clone().numpy()
        else:
            prediction2 = torch.Tensor(np.dot(W_prunned, features.T).T)
        ranks_zaid = self.prediction(prediction2, n_attack, n_traces, interval, key_attack, targets)
        return ranks_zaid,nbre_mask_activated_per_filter

    '''
    Make a consistent system that take in something and manipulate it.
    1.Change from Sat to list of (list of strings for each conjector) for each filter -> Sat_2_StringSat
    2. Manipulate this list of list eg. guess_sat_removed_size
    3. Change it back to Sat ->StringSat_2_Sat
    4. If want to evaulate the guess entropy -> guess_sat 
    '''
    #define a function to remove mask according to size  then we apply guess entropy to see if it affects.
    '''
    Input: 
    sat_exp: a dictionary with keys = filter_num and values = DNF/CNF
    Output: 
    str_exp_lst: list of list , each index of the list corresponds to the filter with the element of a list that is the conjunctor.
    '''
    def Sat_2_StringSat(self, sat_exp, num_filters, mode='dnf'):
        str_exp_lst = []
        new_sat_exp = copy.deepcopy(sat_exp)
        ##Cut the exp_DNF ( each filter). Remove the unwanted size.
        sign = "|"
        sign2 = "&"
        if mode == "cnf":
            sign = "&"
            sign2 = "|"
        for filter in range(num_filters):
            if new_sat_exp[filter] == 1.0 or new_sat_exp[filter] == '1.0':
                str_exp_lst.append(["1.0"])
            elif new_sat_exp[filter] == 0.0 or new_sat_exp[filter] == '0.0':
                str_exp_lst.append(["0.0"])
            else:
                str_exp = str(new_sat_exp[filter])
                mask_per_filter = str_exp.split(sign)
                mask_per_filter_extra = copy.deepcopy(mask_per_filter)
                for mask_index in range(len(mask_per_filter_extra)):
                    mask = mask_per_filter_extra[mask_index]
                    mask = mask.replace(" ", "")
                    if not ("(x" in mask or "(~" in mask):
                        mask = "(" + mask + ")"
                        mask_per_filter[mask_index] = mask
                str_exp_lst.append(mask_per_filter)
        return str_exp_lst

    '''
    Input: 
    str_exp_lst: list of list , each element corresponds to a filter with the element = list of masks/clauses.
    Output: 
    sat_exp: a dictionary with keys = filter_num and values = DNF/CNF
    '''

    def StringSat_2_Sat(self, str_exp_lst, num_filters, mode='dnf'):
        sat_exp = {}
        sign = "|"
        sign2 = "&"
        if mode == "cnf":
            sign = "&"
            sign2 = "|"
        for filter in range(num_filters):
            str_exp = sign.join(str_exp_lst[filter])
            if str_exp == '' or str_exp == '0.0':
                sat_exp[filter] = simplify_logic(0.0, form=mode)
            elif str_exp == '1.0':
                sat_exp[filter] = simplify_logic(1.0, form=mode)
            else:
                sat_exp[filter] = simplify_logic(str_exp, form=mode)
        return sat_exp

    '''
    Input: 
    size_mask_remove = list of int (distinct) that indicate the clause to be removed.
    str_exp_lst: list of list , each index of the list corresponds to the filter with the element of a list that is the clause.
    Output: 
    new_str_sat_exp: list of list, removed_size of the conjunctor according to size_mask_remove
  '''

    def guess_sat_removed_size(self, str_exp_lst, size_mask_remove, num_filters, mode='dnf'):
        new_str_sat_exp = []
        ##Cut the exp_DNF (each filter). Remove the unwanted size.
        sign = "|"
        sign2 = "&"
        if mode == "cnf":
            sign = "&"
            sign2 = "|"
        print("size_mask_remove:", size_mask_remove)
        for filter in range(num_filters):
            new_mask_per_filter = []
            exp_sat = str_exp_lst[filter]
            # print(exp_sat)
            if exp_sat[0] == '1.0' or exp_sat[0] == '0.0':  # maybe set up the 0.0/1.0 as constant.
                new_mask_per_filter.append(exp_sat[0])
            else:
                for mask in exp_sat:
                    mask_variable = mask.split(sign2)
                    if not len(mask_variable) in size_mask_remove:
                        # Remove from the list
                        new_mask_per_filter.append(mask)
                if new_mask_per_filter == []:
                    new_mask_per_filter = ['0.0']
            new_str_sat_exp.append(new_mask_per_filter)
        return new_str_sat_exp
    #define a function to remove mask according to filter they are in/specific mask  then we apply guess entropy to see if it affects.
    '''
    sat_exp_mask: list of string of mask/clause
    filter_remove: list of int for the filter to be not included.
    '''
    def guess_sat_filter_remove(self, str_exp_lst, filter_remove, mode = 'dnf'): ##The sat_exp should be 'dnf'.
        new_str_sat_exp = copy.deepcopy(str_exp_lst)
        num_filters = self.config.layer_2[0]
        for filter in range(num_filters):
            if filter in filter_remove:
                new_str_sat_exp[filter] = ['0.0']
        return new_str_sat_exp


    def guess_sat_variable_remove(self,str_exp_lst, variable_remove_lst, mode = 'dnf'):
        new_str_sat_exp = copy.deepcopy(str_exp_lst)
        num_filters = self.config.layer_2[0]
        unwanted = []
        for filter in range(num_filters):
            for mask in str_exp_lst[filter]:
                for variable in variable_remove_lst:
                    if variable in mask:
                        unwanted.append(mask)
                        new_str_sat_exp[filter].remove(mask)
                        break
        print(unwanted)
        return new_str_sat_exp

    def guess_sat_variable_trim(self,str_exp_lst, variable_remove_lst,num_filters, mode = 'dnf'): #Trim the variable that you dont want. i.e. "x_1 & x_2 & x_3"
        new_str_sat_exp = []
        for filter in range(num_filters):
            new_filter_sat_exp = []
            exp_sat = str_exp_lst[filter]
            # print(exp_sat)
            if exp_sat[0] == '1.0' or exp_sat[0] == '0.0':# maybe set up the 0.0/1.0 as constant.
                new_filter_sat_exp.append(exp_sat[0])
            else:
                for mask in str_exp_lst[filter]:
                    new_mask = mask
                    # print("before:",new_mask)
                    for variable in variable_remove_lst:
                        eq = simplify_logic(mask)

                        if symbols(variable) in eq.free_symbols:
                            # print("filter: ", str(filter))
                            # print("mask:" + new_mask)
                            # print("variable:"+variable)
                            if variable == "x_1":
                                new_mask = new_mask.replace("x_10","y_0")
                                new_mask = new_mask.replace("x_11","y_1")
                                new_mask = new_mask.replace("x_12","y_2")
                            new_mask = new_mask.replace(variable,"")
                            if variable == "x_1":
                                new_mask = new_mask.replace("y_0","x_10")
                                new_mask = new_mask.replace("y_1","x_11")
                                new_mask = new_mask.replace("y_2","x_12")
                            # print("new mask 1:" + new_mask)
                            if "(~ " in new_mask:
                                new_mask = new_mask.replace("(~ ", "( ")
                            elif " ~ " in new_mask:
                                new_mask = new_mask.replace(" ~ ", "  ")
                            elif "~)" in new_mask:
                                new_mask = new_mask.replace("~)", ")")

                            if "&  &" in new_mask: #middle
                                new_mask = new_mask.replace("&  &", "&")
                            elif "( & " in new_mask:
                                new_mask = new_mask.replace("( & ", "(") #beginning
                            elif " & )" in new_mask:
                                new_mask = new_mask.replace(" & )", ")")  # end

                    # print("new mask 2:" + new_mask)
                    if "()" not in new_mask:
                        new_filter_sat_exp.append(new_mask)
                if new_filter_sat_exp == []:
                    new_filter_sat_exp = ['0.0']
            new_str_sat_exp.append(new_filter_sat_exp)
        return new_str_sat_exp


    def guess_sat_mask_include(self,str_exp_lst, mask_include_lst, mode = 'dnf'):
        new_str_sat_exp = []
        num_filters = self.config.layer_2[0]
        for filter in range(num_filters):
            new_mask_per_filter =[]
            for mask in str_exp_lst[filter]:
                if mask in mask_include_lst:
                    new_mask_per_filter.append(mask)
            new_str_sat_exp.append(new_mask_per_filter)
        return new_str_sat_exp

    def guess_sat_mask_remove(self,str_exp_lst, mask_remove_lst, mode = 'dnf'):
        new_str_sat_exp = []
        num_filters = self.config.layer_2[0]
        for filter in range(num_filters):
            new_mask_per_filter =[]
            for mask in str_exp_lst[filter]:
                if not (mask in mask_remove_lst):
                    new_mask_per_filter.append(mask)
            new_str_sat_exp.append(new_mask_per_filter)
        return new_str_sat_exp

    def guess_sat_mask_solely_include_with_variable(self, str_exp_lst, variable_include_lst, num_filters, mode='dnf'):
        new_str_sat_exp = []
        unwanted_str_sat_exp = []
        for filter in range(num_filters):
            new_mask_per_filter = []
            new_unwanted_per_filter = []
            exp_sat = str_exp_lst[filter]
            if exp_sat[0] == '1.0' or exp_sat[0] == '0.0':  # maybe set up the 0.0/1.0 as constant.
                new_mask_per_filter.append(exp_sat[0])
            else:
                for mask in str_exp_lst[filter]:
                    flag = True
                    for variable in variable_include_lst:
                        if variable not in mask:
                            flag = False
                    if flag == False:
                        new_unwanted_per_filter.append(mask)
                    if flag == True:
                        new_mask_per_filter.append(mask)
            if new_mask_per_filter == []:
                new_mask_per_filter = ['0.0']
            if new_unwanted_per_filter == []:
                new_unwanted_per_filter = ['0.0']
            new_str_sat_exp.append(new_mask_per_filter)
            unwanted_str_sat_exp.append(new_unwanted_per_filter)
        return new_str_sat_exp, unwanted_str_sat_exp

    def sat_exp_DNF_Combine(self, str_sat_exp_1, str_sat_exp_2, mode = 'dnf'):
        if str_sat_exp_1 == []:
            for filter in range(len(str_sat_exp_2)):
                str_sat_exp_1.append([])
        # print(len(str_sat_exp_2))
        for filter in range(len(str_sat_exp_2)):
            for mask in str_sat_exp_2[filter]:
                str_sat_exp_1[filter].append(mask)
        # print(str_sat_exp_1)
        return str_sat_exp_1




    # def guess_sat_remove_particular_literals(self, str_exp_lst, literals_lst, mode='dnf'):
    #     new_str_sat_exp = []
    #     unwanted_str_sat_exp = []
    #     num_filters = self.config.layer_2[0]
    #     sympy_lit = []
    #
    #     for filter in range(num_filters):
    #         new_mask_per_filter = []
    #         new_unwanted_per_filter = []
    #
    #         for disjunct in str_exp_lst[filter]:
    #             print(disjunct)
    #             for lit in literals_lst:
    #                 if lit in disjunct:
    #
    #
    #
    #             print(ok)
    #             flag = True
    #             for variable in literals_lst:
    #                 if variable not in mask:
    #                     flag = False
    #             if flag == False:
    #                 new_unwanted_per_filter.append(mask)
    #             if flag == True:
    #                 new_mask_per_filter.append(mask)
    #         new_str_sat_exp.append(new_mask_per_filter)
    #         unwanted_str_sat_exp.append(new_unwanted_per_filter)
    #     return new_str_sat_exp, unwanted_str_sat_exp
    #





