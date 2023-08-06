#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from utils.pyjanitor import auto_toc
toc = auto_toc(row_align='center')

dormers = ['pajamas', 'bus', 'bread', 'lyric', 'paris', 'coin']


def get_proximity():
    """Define and return the proximity matrix as a DataFrame"""
    P = np.array([[1, 3, 2, 2, 1.5, 1.5],
                  [3, 1, 2, 2, 1.5, 1.5],
                  [2, 2, 1, 3, 1.5, 1.5],
                  [2, 2, 3, 1, 1.5, 1.5],
                  [1.5, 1.5, 1.5, 1.5, 1, 2],
                  [1.5, 1.5, 1.5, 1.5, 2, 1]],)
    df_P = pd.DataFrame(P, columns=dormers, index=dormers).applymap('{:.1f}'.format)
    df_P.rename_axis('dormers', inplace=True)
    toc.add_table(df_P, 'Proximity Matrix of Dormers', index=True, preview=False)
    
    return df_P


def plot_proximity(df_P):
    """Plot the graph visualization of proximity matrix"""
    df_P_list = df_P.melt(id_vars='dormers', value_name='weight').astype({'weight': float})
    g_P = nx.from_pandas_edgelist(df_P_list, source='dormers', target='variable', edge_attr=True)
    
    # # Inverse for plotting, layout considers weight as distance not proximity
    # df_P_inv = df_P_list.copy()
    # df_P_inv['weight'] = 1 / df_P_inv['weight']
    # g_P_inv = nx.from_pandas_edgelist(df_P_inv, source='dormers', target='variable', edge_attr=True)
    nx.draw(g_P, pos=nx.spring_layout(g_P), with_labels=True)
    toc.add_fig('Network Illustration of the Dormers\' Proximity',
                caption='The graph illustrates how close the dormers are to each other',
                width=100)

    return g_P
    

def get_cooccurence():
    """Define and return the cooccurence matrix as a DataFrame"""
    O = np.array([[1, 0.7, 0.25, 0.05, 0.5, 0.5],
                  [0.7, 1, 0.25, 0.05, 0.5, 0.5],
                  [0.25, 0.25, 1, 0.05, 0.15, 0.15],
                  [0.05, 0.05, 0.05, 1, 0.05, 0.05],
                  [0.5, 0.5, 0.15, 0.05, 1, 0.7],
                  [0.5, 0.5, 0.15, 0.05, 0.7, 1]],)
    df_O = pd.DataFrame(O, columns=dormers, index=dormers).applymap('{:.2f}'.format)
    df_O.rename_axis('dormers', inplace=True)
    toc.add_table(df_O, 'Co-occurence Matrix of Dormers', index=True, preview=False)
    
    return df_O


def plot_occurence(df_O):
    """Plot the graph visualization of cooccurence matrix"""
    df_O_list = df_O.melt(id_vars='dormers', value_name='weight').astype({'weight': float})
    g_O = nx.from_pandas_edgelist(df_O_list, source='dormers', target='variable', edge_attr=True)
    
    # # Inverse for plotting, layout considers weight as distance not proximity
    # df_O_inv = df_O_list.copy()
    # df_O_inv['weight'] = 1 / df_O_inv['weight']
    # g_O_inv = nx.from_pandas_edgelist(df_O_inv, source='dormers', target='variable', edge_attr=True)
    nx.draw(g_O, pos=nx.spring_layout(g_O), with_labels=True)
    toc.add_fig('Network Illustration of the Dormers\' Proximity',
                caption='The graph illustrates how often the dormers get to interact with to each other'
                ', showing how rarely lyric joins in group dinners',
                width=100)
    
    return g_O
    
    
def get_empathy(df_P, df_O):
    """Compute and return the empathy matrix as a DataFrame"""
    E = np.divide(df_P.set_index('dormers').astype(float).values,
                    df_O.set_index('dormers').astype(float).values)
    df_E = pd.DataFrame(E, columns=dormers, index=dormers).applymap('{:.2f}'.format)
    df_E.rename_axis('dormers', inplace=True)
    toc.add_table(df_E, 'Empathy Matrix of Dormers', index=True, preview=False)
    
    return df_E


def plot_empathy(df_E):
    """Plot the graph visualization of empathy matrix"""
    df_E_list = df_E.melt(id_vars='dormers', value_name='weight').astype({'weight': float})
    g_E = nx.from_pandas_edgelist(df_E_list, source='dormers', target='variable', edge_attr=True)
    
    # Inverse for plotting, layout considers weight as distance not proximity
    # df_E_inv = df_E_list.copy()
    # df_E_inv['weight'] = 1 / df_E_inv['weight']
    # g_E_inv = nx.from_pandas_edgelist(df_E_inv, source='dormers', target='variable', edge_attr=True)
    nx.draw(g_E, pos=nx.spring_layout(g_E), with_labels=True)
    toc.add_fig('Network Illustration of the Dormers\' Empathy',
                caption='The graph illustrates how group tends to consider each other in social gatherings',
                width=100)
    
    return g_E
    

def get_centrality(gs):
    """Compute for the eigenvector centrality scores for each graph"""
    df_central = pd.DataFrame()
    titles = ['Proximity', 'Co-occurence', 'Empathy']
    for g, title in zip(gs, titles):
        df = pd.DataFrame(nx.eigenvector_centrality(g, weight='weight'), index=[title])
        df_central = pd.concat([df_central, df])
        
    df_central = df_central.T.rename_axis('Dormers')
    toc.add_table(df_central, 'Summary Table of Eigenvector Centrality Scores',
                  index=True,
                  preview=False)
    
    return df_central
    

def get_weights(df_E):
    """Compute and return the global empathy weights as a DataFrame"""
    W_raw = df_E.set_index('dormers').values.astype('float')
    W_norm = (W_raw / W_raw.sum(0)).T
    D = np.diagflat(np.diag(W_norm))

    w = (np.linalg.inv(np.eye(W_norm.shape[0]) - W_norm + D)@D).sum(0)
    
    df_w = pd.DataFrame(w,
                        columns=['Global Empathy Weights'],
                        index=dormers)
    df_w.rename_axis('Dormers', inplace=True)
    
    toc.add_table(df_w,
              'Table of Global Emphatic Weights',
              index=True,
              preview=False)
    
    return df_w


def get_modified():
    """Define and return the modified empathy matrix as a DataFrame"""
    M = np.array([[1, 1, 1, 1, 1, 1],
                  [2, 1, 5, 5, 2, 2],
                  [3, 3, 1, 10, 3, 3],
                  [0, 0, 0, 1, 0, 0],
                  [2, 2, 5, 5, 1, 2],
                  [3, 3, 1, 1, 5, 1]])
    df_M = pd.DataFrame(M, columns=dormers, index=dormers).applymap('{:.2f}'.format)
    df_M.rename_axis('dormers', inplace=True)
    toc.add_table(df_M, 'Modified Empathy Matrix of Dormers', index=True, preview=False)
    
    return df_M


def plot_modified(df_M):
    """Plot the graph visualization of modified empathy matrix"""
    df_M_list = df_M.melt(id_vars='dormers', value_name='weight').astype({'weight': float})
    g_M = nx.from_pandas_edgelist(df_M_list, source='dormers',
                                  target='variable',
                                  edge_attr=True,
                                  create_using=nx.DiGraph())
    
    # Inverse for plotting, layout considers weight as distance not proximity
    # df_E_inv = df_E_list.copy()
    # df_E_inv['weight'] = 1 / df_E_inv['weight']
    # g_E_inv = nx.from_pandas_edgelist(df_E_inv, source='dormers', target='variable', edge_attr=True)
    nx.draw(g_M, pos=nx.spring_layout(g_M), with_labels=True)
    toc.add_fig('Network Illustration of the Dormers\' Modified Empathy',
                caption='The graph illustrates how the group, artificially, consider each other',
                width=100)
    
    return g_M
    

def summarize_modifed(df_M, g_M):
    """Compute and return the global empathy weights as a DataFrame"""
    W_raw = df_M.set_index('dormers').values.astype('float')
    W_norm = (W_raw / W_raw.sum(0)).T
    D = np.diagflat(np.diag(W_norm))
    w = (np.linalg.inv(np.eye(W_norm.shape[0]) - W_norm + D)@D).sum(0)
    
    df_w = pd.DataFrame(w,
                        columns=['Global Empathy Weights'],
                        index=dormers)
    
    df = pd.DataFrame().from_dict(nx.eigenvector_centrality(g_M, weight='weight'), orient='index')
    df.columns = ['Eigenvector Centrality']
        
    df_w = pd.merge(df_w, df, right_index=True, left_index=True)
    
    df_w.rename_axis('Dormers', inplace=True)
    toc.add_table(df_w, 'Summary Table for Modified Empathy Matrix',
                  index=True,
                  preview=False)
    return df_w


