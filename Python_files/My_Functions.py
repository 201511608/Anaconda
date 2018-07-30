#####################  #
# CONTENT
# :: 1  # Common_Functions
# :: 2  # UnSupervised
# :: 3  # Reinforcement
# :: 4  # Supervised
# :: 5  #
# :: 6  #
# :: 7  #
# :: 8  #
# :: 9  #
# :: 10 #
#####################  #


# :: 2 
##################################################################################################################
#####################################################################################
# Common_Functions Class
	# :: 1  Array2DataFrame      #Numpy array to DataFrame
	# :: 2  Array2SaveCSV        #Numpy array to DataFrame to SaveFile



class Common_Functions:
    
	def __init__ (self):
		print("Common_Function Class Instance Begin")

	def __del__(self):
		print("Common_Function Class Instance Destroyed")
	
	def check():  # For No Inputs NoNeed Self
		print("In Common_Function Class")

    #Numpy array to DataFrame
	def Array2DataFrame(self,array):
		import numpy as np
		df = pd.DataFrame(data=array)
		return df

	#Numpy array to DataFrame to SaveFile
	def Array2SaveCSV(self, NpArray,filename='Delete'):
		import numpy as np
		df = pd.DataFrame(data=NpArray)
		df.to_csv(filename)
		


		
# :: 2 
##################################################################################################################
#####################################################################################
# UnSupervised Class
	# :: 1   Scatter_Plot             #plots
	# :: 2   Scatter_Plot_WithCg      #plots
	

class Unsupervised:
    
    def __init__( self):
        print("Unsupervised Instance Begin") 
        
    def __del__(self):
       print("Unsupervised Instance Destroyed")
    
    def check():
        print("In Unsupervised class")
    
	# My UnSupervised Learning Functions
	
    #plots
    def Scatter_Plot(self,x,y,labels=[]):
       # labels=[1 for i in range(len(x))]
        plt.scatter(x,y,c=labels,alpha =0.75)
       # plt.scatter(centroids[:,0],centroids[:,1],color='r',marker='D',s=50)
        plt.show()
        
    def Scatter_Plot_WithCg(self,cgx,cgy,x,y,labels=[]):
        # centroids=model.cluster_centers_  # centroids[:,0],centroids[:,1] 
        #labels=[1 for i in range(len(x))]
        plt.scatter(x,y,c=labels,alpha =0.75)
        plt.scatter(cgx,cgy,color='r',marker='D',s=50)
        plt.show()
        
        
    def Inertia_Cluster_Graph(self,Train_Data):
        sm_data=Train_Data
        ks = range(1, 15)
        inertias = []
        for k in ks:
            # Create a KMeans instance with k clusters: model
            modle=KMeans(n_clusters=k)    
            # Fit model to samples
            modle.fit(sm_data)  
            # Append the inertia to the list of inertias
            inertias.append(modle.inertia_)
        plt.plot(ks, inertias, '-o')
        plt.xlabel('Number of Clusters, k')
        plt.ylabel('Inertia')
        plt.xticks(ks)
        plt.show() 
        
        
    def crosstable(predictedData,targetData):
        import pandas as pd
        ct=pd.crosstab(predictedData,targetData)
        print("\nCrossTable\n",ct)
        
    def Normalizer_Kmean(self,Train_Data,Prediction_Data,clusters=5):    
        ##import numpy as np
        from sklearn.preprocessing import Normalizer
        from sklearn.cluster import KMeans

        # Market_Data=pd.read_csv('sharemarket_data.csv',header=0,usecols= [x for x in range(1,963)])
        ##companiessm_data=np.loadtxt('sharemarket_data.txt') #sharemarket_data
        ##companies=pd.read_csv('sharemarket_data.csv',header=0,usecols= [0]).values.reshape(60)

        #7.1
        normalizer= Normalizer() # Normalizing the Data
        kmeans= KMeans(n_clusters=clusters)   # Change Cluster and check the results
        pipeline = make_pipeline(normalizer,kmeans)
        pipeline.fit(Train_Data)

        #7.2
        prediction =pipeline.predict(Prediction_Data)
        return prediction
    
    def StandardScaler_Kmean(self,Train_Data,Prediction_Data,clusters=5):                     
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            # StandardScaler  MaxAbsScalar   Normilizer
            scaler = StandardScaler()
            # Create KMeans instance: kmeans
            kmeans = KMeans(n_clusters=clusters)


            # Create pipeline: pipeline
            pipeline = make_pipeline(scaler,kmeans)  # Make_pipeline
            # Fit the pipeline to samples
            pipeline.fit(samples)
            # Calculate the cluster labels: labels
            labels = pipeline.predict(samples)
            prediction=labels
            return prediction
        
        
    def Dendgrogram_Plot(self,TrainData,labels,method='complete'):
        #label=[1 for i in range(len(TrainData))]
        
        from scipy.cluster.hierarchy import linkage, dendrogram
        import matplotlib.pyplot as plt

        # Calculate the linkage: mergings:: This performs hierarchical clustering
        mergings = linkage(TrainData, method=method)

        # Plot the dendrogram
        dendrogram(mergings,labels=label,leaf_rotation=90,leaf_font_size=6)
        plt.show()




    def DendgrogramDistance_SubPlot(self,TrainData,labels,distance=5,method='complete'):
        from scipy.cluster.hierarchy import linkage
        from scipy.cluster.hierarchy import fcluster

        samples=TrainData
        varieties_=labels
       # varieties=['Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Kama wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Rosa wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat', 'Canadian wheat']

        #Measures distance.
        mergings = linkage(samples,method =method)  # Using above samples

        dendrogram(mergings,labels=varieties_,leaf_rotation=90,leaf_font_size=6,)
        print("Full Dendrogram, No distance limits")
        plt.show()

        #Extracting Intermediate Clusters!
        labels_f = fcluster(mergings,distance,criterion='distance')
        print (labels)
        print ("At distance",distance,"Division considered Above this all assign as Indivudial Number")
        dendrogram(mergings,labels=labels_f,leaf_rotation=90,leaf_font_size=6,)
        plt.show()



    def AllPlots(self,samples,labels,distance):
        
        #Inertia Plot
        Unsupervised.Inertia_Cluster_Graph(self,samples)

        #Scatter Plot                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              +
        Unsupervised.Scatter_Plot(self,samples[:,0],samples[:,1],labels)
        Unsupervised.Scatter_Plot(self,samples[:,2],samples[:,3],labels)

        #DendroGram Plot
        Unsupervised.DendgrogramDistance_SubPlot(self,samples,labels,distance=distance,method='complete')
    
    def Tsne(self,samples,labels,learning_rate=100):
        from sklearn.manifold import TSNE
        model = TSNE(learning_rate=learning_rate)
        tsne_features = model.fit_transform(samples)
        return tsne_features
    def TsnePlot(self,samples,labels_numbers,learning_rate=100):
        print("Sample input Ex :",samples[0,:])
        tsne_features=Unsupervised.Tsne(self,samples,labels,learning_rate=100)
        xs = tsne_features[:,0]
        ys = tsne_features[:,1]
        print("Tsne Output Ex:",tsne_features[0,:])
        plt.scatter(xs, ys, c=labels_numbers)
        plt.show()
    def TsnePlotLabel(self,samples,labels_numbers,labels_name,learning_rate=100):       
        tsne_features=Unsupervised.Tsne(self,samples,labels,learning_rate=100)
        xs = tsne_features[:,0]
        ys = tsne_features[:,1]
        print("Sample input Ex :",samples[0,:])
        print("Tsne Output Ex:",tsne_features[0,:])
        plt.scatter(xs, ys, c=labels_numbers)
        for x, y, company in zip(xs, ys, companies):
            plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
        plt.show()
    
    def Pca_DimensionReduce(self,TrainData,n_components=2):
        #18
        #PCA fit and Transforms  ....  Don't have Predict  . It just Reduce the Dimensions!
        from sklearn.decomposition import PCA

        # Create a PCA model with 2 components: pca
        pca = PCA(n_components=n_components)   # n_components = Components Require !!!

        # Fit the PCA instance to the scaled samples
        pca.fit(TrainData)

        # Transform the scaled samples: pca_features
        pca_features = pca.transform(TrainData)
        
        
        print("Just Reduces the Dimension",TrainData.shape,"->",pca_features.shape)

        return pca_features
    
        
    def PcaPlot(self,TrainData):
        grains=TrainData
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        model=PCA()
        model.fit(grains)
        transformed=model.transform(grains)
        ## we can use both to gether  FIT AND TRANSFORM
        #  model.fit_transform(grains)
        #14.1
        print("Actual Data")
        plt.scatter(grains[:,0], grains[:,1])
        #plt.axis('equal')
        plt.show()

        #14.2
        print("\n PCA Transformed \n Data Shift to Mean=0  and Rotates")
        plt.scatter(transformed[:,0], transformed[:,1])
        plt.axis('equal')
        plt.show()

        #print
        print("\n\nActual Data :",grains[0,:])
        print("PCA Transformed Data :",transformed[0,:])
        print("Mean of NotTransformed Data:",model.mean_)
        
        
        mean = model.mean_
        print("\n\nmean :",mean)

        # Get the first principal component: first_pc
        first_pc = model.components_[0,:]
        print("first_pc",first_pc)
        print("The first principal component of the data is the direction in which the data varies the most")
        # Plot first_pc as an arrow, starting at mean
        plt.scatter(grains[:,0], grains[:,1])
        plt.arrow(mean[0],mean[1], first_pc[0], first_pc[1], color='red', width=0.051)
        plt.show()
        
        
    def StandardScalarPca_VariencePlot(self,TrainData):
        #16  PCA PLOT
        # Perform the necessary imports
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        import matplotlib.pyplot as plt

        fish_Data=TrainData

        # Create scaler: scaler
        scaler = StandardScaler()

        # Create a PCA instance: pca
        pca = PCA()

        # Create pipeline: pipeline
        pipeline = make_pipeline(scaler,pca)

        # Fit the pipeline to 'samples'
        pipeline.fit(fish_Data)

        # Plot the explained variances
        features =range(pca.n_components_)    ## HIGHLIGHTING
        print("Features is equal to Input :",features)
        plt.bar(features,pca.explained_variance_)
        plt.xlabel('PCA feature')
        plt.ylabel('variance')
        plt.xticks(features)
        plt.show()
        print(" Plot Shows to what Level Dimension Can be reduce\n Higher the Varience value higher the Results dependence ")
        print (" Select PCA(n_components = Base up on above Graph)")
        
    def Tfidf_ConverttoCsr(self,TrainData):
        #19 
        # return csr_mat
        # tfidf
        # tfidf -> It transforms a list of documents into a word frequency array, which it outputs as a csr_matri
        print("tfidf -> converts to Csr_mat")
        print("csr matrix reduse the space by remembering only non zero entries \n")
        documents= TrainData
        # Import TfidfVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Create a TfidfVectorizer: tfidf
        tfidf = TfidfVectorizer() 

        # Apply fit_transform to document: csr_mat
        csr_mat = tfidf.fit_transform(documents)

        # Print 

        words = tfidf.get_feature_names()
        print()
        print(documents)
        print()
        print("\nLength :",len(words))
        print(words)
        print()
        print("csr_mat.toarray :\n",csr_mat.toarray())
        print()
        print("Csr_Matrix :\n",csr_mat)

        
        # return
        return csr_mat
    
    
    def TruncatedSvd_Kmean(self,TrainData_csr_matrix,titles,Svd_components=50,n_clusters=6):
        #Svd_components -> Dimension Reduction
        #n_clusters -> n_clusters Reducion
        from sklearn.decomposition import TruncatedSVD
        from sklearn.cluster import KMeans
        from sklearn.pipeline import make_pipeline

        # Create a TruncatedSVD instance: svd
        svd = TruncatedSVD(n_components=Svd_components)

        # Create a KMeans instance: kmeans
        kmeans =KMeans(n_clusters=n_clusters)   #

        # Create a pipeline: pipeline
        pipeline = make_pipeline(svd,kmeans)
        
        
        import pandas as pd
        
        articles = TrainData_csr_matrix         # Dirct csr matrix
        titles = titles

       
        import pandas as pd

        # Fit the pipeline to articles
        pipeline.fit(articles)

        # Calculate the cluster labels: labels
        labels = pipeline.predict(articles)
        print(len(labels))
        print("Labels",labels)

        # Create a DataFrame aligning labels and titles: df
        df1 = pd.DataFrame({'label': labels, 'article': titles})


        # Print
        print(df1.sort_values('label'))

    def Nmf(self,TrainData_csr_matrix,titles,n_components=6):
        #21
        # NMF   :: Non Negative Matrix Factorization (NMF)   #for Non Negative
        #  Dimension Reduction technique 
        from sklearn.decomposition import NMF

        #df = pd.read_csv('wikipedia-vectors.csv', index_col=0)
        articles =TrainData_csr_matrix #csr_matrix(df.transpose())         # Dirct csr matrix
        titles =titles  #list(df.columns)

        # Create an NMF instance: model
        model = NMF(n_components=n_components)

        # Fit the model to articles
        model.fit(articles) #articles is CSR_MATRIX

        # Transform the articles: nmf_features
        nmf_features = model.transform(articles)

        # Print the NMF features
        print(nmf_features)

        
        
        # Create a pandas DataFrame: df
        df = pd.DataFrame(nmf_features,index=titles)

        # Print the row for 'Anne Hathaway'
        print(df.iloc[0])

        # Print the row for 'Denzel Washington'
        print(df.iloc[1])
        print("Model.Components_\n",model.components_)
        
    def Nmf_CompImshow(self,TrainData,n_components=7):
       
        samples=TrainData
        # Import NMF
        from sklearn.decomposition import NMF

        # Create an NMF model: model
        model = NMF(n_components=n_components)

        # Apply fit_transform to samples: features
        features = model.fit_transform(samples)

        # Call show_as_image on each component
        for component in model.components_:
            Unsupervised.show_as_image(self,component)

        # Select the 0th row of features: digit_features
        digit_features = features[0,:]

        # Print digit_features
        print(digit_features)

    
    def show_as_image(self,sample):
        from matplotlib import pyplot as plt
        bitmap = sample.reshape((13, 8))
        plt.figure()
        plt.imshow(bitmap, cmap='gray', interpolation='nearest')
        plt.colorbar()
        plt.show()
        
    def Lcd_Imshow(self,samples,x=13,y=8):
        samples = samples #pd.read_csv('lcd-digits.csv',header=None).values  ## IMPORTANT
        # Import pyplot
        from matplotlib import pyplot as plt

        # Select the 0th row: digit
        digit = samples[10,:]

        # Print digit
        print(digit)

        # Reshape digit to a 13x8 array: bitmap
        bitmap = digit.reshape(x,y)

        # Print bitmap
        print(bitmap)

        # Use plt.imshow to display bitmap
        plt.imshow(bitmap, cmap='gray', interpolation='nearest')
        plt.colorbar()
        plt.show()
    
    def NmfNormalizeDot(self,TrainData_Csr,Article_index=0):
        #29
        # give similar article types
        #29.1
        articles=TrainData
        from sklearn.decomposition import NMF

        # Create an NMF instance: model
        model = NMF(n_components=6)

        # Fit the model to articles
        model.fit(articles)

        # Transform the articles: nmf_features
        nmf_features = model.transform(articles)

        #
        # Perform the necessary imports
        import pandas as pd
        from sklearn.preprocessing import normalize

        # Normalize the NMF features: norm_features
        norm_features =normalize(nmf_features)
        # Create a DataFrame: df
        df =pd.DataFrame(norm_features,titles)

        # Select the row corresponding to 'Cristiano Ronaldo': article
        article = df.iloc[Article_index]     # TO  GET SIMILAR TYPE OF ARTICLES !!!

        # Compute the dot products: similarities
        similarities = df.dot(article)

        # Display those with the largest cosine similarity
        print(similarities.nlargest())
 

 
    
# :: 3
##################################################################################################################
#####################################################################################
# Reinforcement Class
	# :: 1  Reinforcement_PoleBalance  
	# :: 2   
 

class Reinforcement:
    def __init__( self):
        print("Reinforcement Class Instance Begin ") 

    def __del__(self):
	    print("Reinforcement Class Instance Destroyed")

    def check():
	    print("In Reinforcement class")
		
	### PoleBalance	
    class Reinforcement_PoleBalance:
        # Defautl Goal_steps , LR, score_requirement and initia_games Defined  
        def __init__( self,goal_steps = 500,LR = 1e-3,score_requirement = 100,initial_games = 10000):
            print("Reinforcement_PoleBalance Class Instance Begin :: Takes default inputs") 
            self.LR = LR
            self.goal_steps = goal_steps
            self.score_requirement = score_requirement
            self.initial_games = initial_games 


        def __del__(self):
            print("Reinforcement_PoleBalance Class Instance Destroyed")

        def check():
            print("In Reinforcement_PoleBalance class")

         # 1.1
        # Pole Balance
        # Reinforcement 
        # Make it Better
        def Pole_Balance(self):
            import gym
            import random
            import numpy as np
            import tflearn
            from tflearn.layers.core import input_data, dropout, fully_connected
            from tflearn.layers.estimator import regression
            from statistics import median, mean
            from collections import Counter

            LR = 1e-3
            env = gym.make("CartPole-v0")
            env.reset()
            goal_steps = 500
            score_requirement = 100
            initial_games = 10000
            return score_requirement

        def some_random_games_first(self):
            # Each of these is its own game.
            for episode in range(5):
                env.reset()
                # this is each frame, up to 200...but we wont make it that far.
                for t in range(200):
                    # This will display the environment
                    # Only display if you really want to see it.
                    # Takes much longer to display it.
                    env.render()

                    # This will just create a sample action in any environment.

    # In this environment, the action can be 0 or 1, which is left or right
                    action = env.action_space.sample()

                    # this executes the environment with an action, 
                    # and returns the observation of the environment, 
                    # the reward, if the env is over, and other info.
                    observation, reward, done, info = env.step(action)
                    if done:
                        break

        #some_random_games_first()
        def initial_population(self):
            import numpy as np
            import random
            import gym
            from statistics import median, mean
            from collections import Counter

            env = gym.make("CartPole-v0")
            env.reset()

            # [OBS, MOVES]
            training_data = []
            # all scores:
            scores = []
            # just the scores that met our threshold:
            accepted_scores = []
            # iterate through however many games we want:
            for _ in range(self.initial_games):
                score = 0
                # moves specifically from this environment:
                game_memory = []
                # previous observation that we saw
                prev_observation = []
                # for each frame in 200
                for _ in range(self.goal_steps):
                    # choose random action (0 or 1)
                    action = random.randrange(0,2)
                    # do it!
                    observation, reward, done, info = env.step(action)

                    # notice that the observation is returned FROM the action
                    # so we'll store the previous observation here, pairing
                    # the prev observation to the action we'll take.
                    if len(prev_observation) > 0 :
                        game_memory.append([prev_observation, action])
                    prev_observation = observation
                    score+=reward
                    if done: break

                # IF our score is higher than our threshold, we'd like to save
                # every move we made
                # NOTE the reinforcement methodology here. 
                # all we're doing is reinforcing the score, we're not trying 
                # to influence the machine in any way as to HOW that score is 
                # reached.
                if score >= self.score_requirement:
                    accepted_scores.append(score)
                    for data in game_memory:
                        # convert to one-hot (this is the output layer for our neural network)
                        if data[1] == 1:
                            output = [0,1]
                        elif data[1] == 0:
                            output = [1,0]

                        # saving our training data
                        training_data.append([data[0], output])

                # reset env to play again
                env.reset()
                # save overall scores
                scores.append(score)

            # just in case you wanted to reference later
            training_data_save = np.array(training_data)
            np.save('saved1.npy',training_data_save)

            # some stats here, to further illustrate the neural network magic!
            print('Average accepted score:',mean(accepted_scores))
            print('Median score for accepted scores:',median(accepted_scores))
            print(Counter(accepted_scores))

            return training_data



        def neural_network_model(self, input_size):
            import tflearn
            from tflearn.layers.core import input_data, dropout, fully_connected
            from tflearn.layers.estimator import regression

            network = input_data(shape=[None, input_size, 1], name='input')

            network = fully_connected(network, 128, activation='relu')
            network = dropout(network, 0.8)

            network = fully_connected(network, 256, activation='relu')
            network = dropout(network, 0.8)

            network = fully_connected(network, 512, activation='relu')
            network = dropout(network, 0.8)

            network = fully_connected(network, 512, activation='relu')
            network = dropout(network, 0.8)

            network = fully_connected(network, 512, activation='relu')
            network = dropout(network, 0.8)

            network = fully_connected(network, 2, activation='softmax')
            network = regression(network, optimizer='adam', learning_rate=self.LR, loss='categorical_crossentropy', name='targets')
            model = tflearn.DNN(network, tensorboard_dir='log')

            return model
        
        def neural_network_model_initialize (self,training_data):
            import numpy as np
            X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
            y = [i[1] for i in training_data]
            model = Reinforcement.Reinforcement_PoleBalance.neural_network_model(self,input_size = len(X[0]))
            return model

        def train_model(self,training_data, model):
            import numpy as np
        
            X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
            y = [i[1] for i in training_data]
            model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openai_learning')
            return model

        # model = train_model(initial_population())

        def Pole_test(self,model):
            # 1.2
            import gym
            import random
            import numpy as np
            import tflearn
            from tflearn.layers.core import input_data, dropout, fully_connected
            from tflearn.layers.estimator import regression
            from statistics import median, mean
            from collections import Counter

            LR = 1e-3
            env = gym.make("CartPole-v0")
            env.reset()
            goal_steps = 500
            score_requirement = 50
            initial_games = 10000
        #    def     training_data =initial_population()
        #         model = train_model(training_data)
            scores = []
            choices = []

            for each_game in range(10):
                score = 0
                game_memory = []
                prev_obs = []
                env.reset()
                for _ in range(goal_steps):
                    env.render()
                    if len(prev_obs)==0:
                        action = random.randrange(0,2)
                    else:
                        action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
                    choices.append(action)               
                    new_observation, reward, done, info = env.step(action)
                    prev_obs = new_observation
                    game_memory.append([new_observation, action])
                    score+=reward
                    if done: break
                scores.append(score)
            env.close()
            print('Average Score:',sum(scores)/len(scores))
            print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
            print(score_requirement)
        
#       #  Function Call ::  Reinforcement_PoleBalance
#       #  r=Reinforcement.Reinforcement_PoleBalance()
#       #  training_data=r.initial_population()
#       #  model = r.neural_network_model_initialize (training_data)
#       #  model=r.train_model(training_data,model)
#       #  r.Pole_test(model)        









# ::4 
##################################################################################################################
#####################################################################################
# Supervised Class
    # :: 1   
    # :: 2   


class Supervised:

    def __init__( self):
        print("Supervised Instance Begin") 

    def __del__(self):
        print("Supervised Instance Destroyed")

    def check():
        print("In Supervised class")

    # My Supervised Learning Functions

    def Exploratory_Data_Analysis_():
        #3
        #Example
        # Iris Data -> Print & Visualise
        # Exploratory data analysis (EDA)
        from sklearn import datasets                       #Import Datasets  #Check different types of data sets with in it !
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        iris = datasets.load_iris()

        #print
        print("type(iris) :",type(iris))  # -Bunch -> Similar to Dictonary
        print("\niris.keys() :",iris.keys())
        print("\ntype(iris.data) :",type(iris.data),"type(iris.target) :",type(iris.target))
        print("\niris.data.shape :",iris.data.shape)                                # petal Length and width  sepal length and width
        print("\niris.target_names :",iris.target_names)                            # 0 1 2  ->  setosa versicolor virginica

        x=iris.data
        y=iris.target
        print("x  :",type(x))
        df=pd.DataFrame(x, columns = iris.feature_names)    # Create DataFrame

        #print
        print('\ndf.head() :\n',df.head())
        print('\ndf.info() :\n',df.info())
        print('\ndf.describe() :\n',df.describe())

        #visual Data
        #c=y -> Colour by species
        _=pd.scatter_matrix(df,c=y,figsize = [10,10],s=150,marker ='D')   # Visual Plot                     !Important!
        plt.show()                                                        # Histogram and Scatter Plot


    def Exploratory_Data_Analysis(df_X,df_Y):
        #3
        #Example
        # Iris Data -> Print & Visualise
        # Exploratory data analysis (EDA)
        from sklearn import datasets                       #Import Datasets  #Check different types of data sets with in it !
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        #
        ##
        # print
        print('\ndf.head() :\n',df.head())
        print('\ndf.info() :\n',df.info())
        print('\ndf.describe() :\n',df.describe())

        #visual Data
        #c=y -> Colour by species
        _=pd.scatter_matrix(df_X,c=df_Y,figsize = [10,10],s=150,marker ='D')   # Visual Plot                     !Important!
        plt.show()             

        ## Ex
        # import numpy as np
        # import pandas as pd
        # df =pd.read_csv('Data_files/US_Voting_Filtered_Data_.csv')
        # Exploratory_Data_Analysis(df.drop(['Unnamed: 0','diabetes'],axis =1),df['diabetes'])
    def SeaBorn_Plot_():
    #  4.2
    #  seaborn plot library
    #  Example
        import seaborn as sns
        import matplotlib.pyplot as plt

        import pandas as pd
        df=pd.read_csv('Data_files/US_Voting_Data.csv')

        plt.figure(1)
        sns.countplot(x='education', hue='party', data=df, palette='RdBu')
        plt.xticks([0,1], ['No', 'Yes'])
        print(" Democrats voted resoundingly against this bill, compared to Republicans")
        plt.show()

        plt.figure(2)
        sns.countplot(x='satellite', hue='party', data=df, palette='RdBu')
        plt.xticks([0,1], ['No', 'Yes'])
        print(" Democrats voted in favor of  'satellite")
        plt.show()

        plt.figure()
        sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
        plt.xticks([0,1], ['No', 'Yes'])
        print(" Democrats voted in favor of 'missile'")
        plt.show()

        plt.figure()
        sns.countplot(x='infants', hue='party', data=df, palette='RdBu')
        plt.xticks([0,1], ['No', 'Yes'])
        plt.show()

        plt.figure()
        sns.countplot(x='water', hue='party', data=df, palette='RdBu')
        plt.xticks([0,1], ['No', 'Yes'])
        plt.show()


        plt.figure()
        sns.countplot(x='religious', hue='party', data=df, palette='RdBu')
        plt.xticks([0,1], ['No', 'Yes'])
        plt.show()     

    def SeaBorn_Plot(df,columnName_1,columnName_2,No_label='No',Yes_label='Yes'):
        #
        #  4.2
        #  seaborn plot library
        #  Example
        import seaborn as sns
        import matplotlib.pyplot as plt

        import pandas as pd
        df=df

        plt.figure(1)
        sns.countplot(x=columnName_1, hue=columnName_2, data=df, palette='RdBu')
        plt.xticks([0,1], [No_label, Yes_label])
        plt.show()

        ## EX:
        # df=pd.read_csv('Data_files/US_Voting_Data.csv')
        # Supervised.SeaBorn_Plot(df,'education','party')
    def KNearestNeighbors_():
        # 5
        # Classification
        # K-Nearest Neighbors
                # k 5 -> max of 5 points
                # Make a Sets of Decision Bountaries
                # Higer n_neighbors Low complex   Less Smoot Boundary
                # Lower n_neighbors High Complex  High Smoot Boundary
        # On iris data
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import datasets                       #Import Datasets  

        iris = datasets.load_iris()

        knn= KNeighborsClassifier(n_neighbors=6)

        knn.fit(iris['data'],iris['target'])   # inputs as Numpy/pandasDataFrame

        prediction = knn.predict(iris['data'])

        print('Prediction :',prediction)
        print('target',iris.target)

    def KNearestNeighbors(x_train,y_target):
        # 6
        # Classification
        # K-Nearest Neighbors
                # k 5 -> max of 5 points
                # Make a Sets of Decision Bountaries
                # Higer n_neighbors Low complex   Less Smoot Boundary
                # Lower n_neighbors High Complex  High Smoot Boundary
        # On iris data
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import datasets                       #Import Datasets  

        iris = datasets.load_iris()

        knn= KNeighborsClassifier(n_neighbors=n)

        knn.fit(x_train,y_target)   # inputs as Numpy/pandasDataFrame

        prediction = knn.predict(x_train)

        print('Prediction :',prediction)
        print('target',y_target)


    def MNIST():
        #9
        #MNIST
        #Redused version of data is obtain from Sklearn



        from sklearn import datasets
        digits = datasets.load_digits()

        print("digits.key  :",digits.keys())
        print("\ndigits.DESCR  :",digits.DESCR)
        print("\ndigits.images.shape  :",digits.images.shape)
        print("\ndigits.data.shape   :",digits.data.shape)

        import matplotlib.pyplot as plt
        plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()
        #9.1
        # Create feature and target arrays
        X = digits.data
        y = digits.target

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)


        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=7)

        # Fit
        knn.fit(X_train, y_train)

        # accuracy
        print("Score : ",knn.score(X_test,y_test))

    def Overfitting_Underfitting_():
        # 10
        # Overfitting and underfitting
        # Setup arrays to store train and test accuracies
        import numpy as np
        neighbors = np.arange(1, 10)
        train_accuracy = np.empty(len(neighbors))
        test_accuracy = np.empty(len(neighbors))

        # Loop over different values of k
        for i, k in enumerate(neighbors):

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train,y_train)

            #Accuracy on the training set
            train_accuracy[i] = knn.score(X_train, y_train)
            #Accuracy on the testing set
            test_accuracy[i] = knn.score(X_test, y_test)

        #Plot
        import matplotlib.pyplot as plt
        plt.title('k-NN: Varying Number of Neighbors')
        plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
        plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
        plt.legend()
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Accuracy')
        print("        Left Overfitting --- Right underfitting")
        plt.show()




    def LinearRegression_():
        #12
        #Linear Model
        #Regression
        # House Data Train Base up on Room data only !
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        boston = pd.read_csv('data_files/Boston_House_Data.csv')
        X= boston.drop('MEDV', axis =1).values
        y= boston.MEDV.values
        # RM room features only for training  # Not all features for trianing  
        X_rooms = X[:,5]  #i.e. RM
        y = y.reshape(-1,1)
        X_rooms= X_rooms.reshape(-1,1)


        from sklearn import linear_model
        reg = linear_model.LinearRegression()

        reg.fit(X_rooms, y)
        prediction_space = np.linspace(min(X_rooms),max(X_rooms)).reshape(-1, 1) # Min to Max

        plt.scatter(X_rooms,y,color='blue')
        plt.plot(prediction_space,reg.predict(prediction_space),color='black',linewidth=3) # [line]to show min to max room to price
        plt.show()
        print("Score :",reg.score(X_rooms,y))

    def LinearREgression_CVSplit_():

        #18
        # Cross Validation  (CV)    #the more computationally expensive cross-validation becomes
        # Fold  : Split into folds   1 2 3 4 5     
        # 1 Test    2 3 4 5 Train
        # 2 Test    1 3 4 5 Train

        import pandas as pd
        df = pd.read_csv('data_files/Gapminder_Region_data.csv')
        y = df.life.values.reshape(-1,1)
        X= df.drop(["life","Region"],axis=1).values


        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()

        from sklearn.model_selection import cross_val_score
        cv_results = cross_val_score(reg, X,  y, cv=5)     # Return Cross Validation Score
        print("Score :",cv_results)
        import numpy as np
        print("Mean :",np.mean(cv_results))
    def RidgeRegressoin_Split_():

        #21
        # Ridge Regression 
        # lOSS FUNCTION    [OLS+(ALPHA)*ai^2]         alpha = 0 overfitting  alpha =inf   underfitting
        # Alpha controls model complexity

        import pandas as pd
        df = pd.read_csv('data_files/Gapminder_Region_data.csv')
        y = df.life.values.reshape(-1,1)
        X= df.drop(["life","Region"],axis=1).values



        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=0.1,normalize=True)   # (alpha and normalize ?)
        ridge.fit(X_train, y_train)
        ridge_pred=ridge.predict(X_test)
        ridge.score(X_test, y_test)


    def Reshaping_():
        # 13
        # reshaping !!
        # Gapminder_Region_data  
        # to predict life expectensy for different region 

        #-> Traindata [ []  []  []  [] ]   -> Target  [  []  []  [] ]     || ID to ID  ||
        import numpy as np
        import pandas as pd


        df = pd.read_csv('data_files/Gapminder_Region_data.csv')


        y = df.life.values
        X = df.fertility.values
        tem=y
        print("Dimensions of y before reshaping: {}".format(y.shape))
        print("Dimensions of X before reshaping: {}".format(X.shape))

        # Reshape X and y
        y = y.reshape(-1,1)
        X = X.reshape(-1,1)

        print("Dimensions of y after reshaping: {}".format(y.shape))
        print("Dimensions of X after reshaping: {}".format(X.shape))
        print(tem)
        print(y)

    def HeatMap_():
        # 14
        # Gapminder Data
        # Heat Map:
        # Green show positive correlation, Red show negative correlation
        import seaborn as sns
        import pandas as pd

        df = pd.read_csv('Gapminder_Region_data.csv')

        # Heat Map Plot
        plt.figure(1)
        sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
        print("Green show positive correlation, Red show negative correlation")
        plt.show()
        print(df.head(),df.info(),df.describe())

    def LassoRegression_():
        #22
        # Lasso Regression 
        # lOSS FUNCTION    [OLS+(ALPHA)*|ai|]         alpha = 0 overfitting  alpha =inf   underfitting
        # Alpha controls model complexity
        # Can select important features of dataset : shrinks less important data to almost 0


        import pandas as pd
        df = pd.read_csv('data_files/Gapminder_Region_data.csv')
        y = df.life.values.reshape(-1,1)
        X= df.drop(["life","Region"],axis=1).values



        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.1,normalize=True)   # (alpha and normalize ?)
        lasso.fit(X_train, y_train)
        lasso_pred=ridge.predict(X_test)
        lasso.score(X_test, y_test)     
    def LassoRegression_ImpFeatures_():
        #22
        # Lasso Regression 
        # lOSS FUNCTION    [OLS+(ALPHA)*|ai|]         alpha = 0 overfitting  alpha =inf   underfitting
        # Alpha controls model complexity
        # Can select important features of dataset : shrinks less important data to almost 0


        import pandas as pd
        df = pd.read_csv('data_files/Gapminder_Region_data.csv')
        y = df.life.values.reshape(-1,1)
        X= df.drop(["life","Region"],axis=1).values



        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.1,normalize=True)   # (alpha and normalize ?)

        # 23
        # Using Lasso .coef_  
        # Can Determinte which is importan features
        lasso_coef = lasso.fit(X_train,y_train).coef_
        names= df.drop(["life","Region"],axis=1).columns
        import matplotlib.pyplot as plt
        _=plt.plot(range(len(names)), lasso_coef)
        _=plt.xticks(range(len(names)),names, rotation=60)
        _=plt.ylabel('Coefficients')
        print(':: Most Important Features ::')
        plt.margins(0.1)  # check this ?
        plt.show()


    def RidgeRegression_Alpha_():
        # 25 Ridge
        #    L1 regularization  + Alpha = LossCalculation
        # practice fitting ridge regression models over a range of different alphas, and plot cross-validated R2 scores for each
        # cross-validation scores change with different alphas
        # Loss function to show
        def display_plot(cv_scores, cv_scores_std):  # FillBetween plot
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(alpha_space, cv_scores)
            std_error = cv_scores_std / np.sqrt(10)
            ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2) # Loss function to show
            ax.set_ylabel('CV Score +/- Std Error')
            ax.set_xlabel('Alpha')
            ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
            ax.set_xlim([alpha_space[0], alpha_space[-1]])
            ax.set_xscale('log')
            print("Mean - StandardDeviation: How Spread the value.")
            plt.show()

        import pandas as pd
        df = pd.read_csv('data_files/Gapminder_Region_data.csv')
        y = df.life.values.reshape(-1,1)
        X= df.drop(["life","Region"],axis=1).values


        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score
        import numpy as np


        alpha_space = np.logspace(-4, 0, 50)                     # Setup the array of alphas and lists to store scores
        ridge_scores = []
        ridge_scores_std = []

        ridge = Ridge(normalize=True)                            # Create a ridge regressor: ridge


        for alpha in alpha_space:                                # Compute scores over range of alphas
            ridge.alpha = alpha                                  # Specify the alpha value to use: ridge.alpha
            ridge_cv_scores = cross_val_score(ridge, X,y,cv=10 ) # Perform 10-fold CV: ridge_cv_scores
            ridge_scores.append(np.mean(ridge_cv_scores))        # Append the mean of ridge_cv_scores to ridge_scores
            ridge_scores_std.append(np.std(ridge_cv_scores))     # Append the std of ridge_cv_scores to ridge_scores_std
                                                                 # StandardDeviation : 
        display_plot(ridge_scores, ridge_scores_std)             # Display the plot



    def Confusion_Matrix():
        # 26
        # Confusion Matrix
        # Classification_report
        # Knn Classifier :: Split Data into Training set and Test set
        # unit 3

        import pandas as pd
        df=pd.read_csv('data_files/US_Voting_Filtered_Data_.csv')  
        X = df.drop('party', axis=1).values
        y = df['party'].values

        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)


        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=8) 
        knn.fit(X_train,y_train)
        predict=knn.predict(X_test)

        #Confusion Matrix
        from sklearn.metrics import confusion_matrix
        print("Confusion Matrix :\n[TP, FN]\n[FP, TN]\n",confusion_matrix(y_test,predict))  
        ## [TP, FN]  [0=0, 0=1]
        ## [FP, TN]  [1=0, 1=1]

        #Classification Report
        from sklearn.metrics import classification_report
        print("\nprecision    :For True: Do not say as False")
        print("recall       :For False: Say as False\n")
        print(classification_report(y_test, predict))

        # Accuracy Score :: not always an informative metric
        print("Score: ",knn.score(X_test,y_test))

    def KNearestNeighbor_ConFusion_():
        # 27
        # KNN 
        # Diabatic Data # PIMA Indians # Pima Indians Diabetes
        # # Confusion Matrix
        # # Classification_report



        import pandas as pd
        df=pd.read_csv('data_files/Diabetes_Data.csv')  
        df.insulin.replace((0), (155.54), inplace=True)
        X = df.drop('diabetes', axis=1).values
        y = df['diabetes'].values

        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.4,random_state=42)


        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=8) 
        knn.fit(X_train,y_train)
        predict=knn.predict(X_test)

        #Confusion Matrix
        from sklearn.metrics import confusion_matrix
        print("Confusion Matrix :\n[TP, FN]\n[FP, TN]\n",confusion_matrix(y_test,predict))  
        ## [TP, FN]  [0=0, 0=1]
        ## [FP, TN]  [1=0, 1=1]

        #Classification Report
        from sklearn.metrics import classification_report
        print("\nprecision    :For True: Do not say as False")
        print("recall       :For False: Say as False\n")
        print(classification_report(y_test, predict))

        # Accuracy Score :: not always an informative metric
        print("Score: ",knn.score(X_test,y_test))

    def LogisticRegression_Classif_():
        # 28
        #  Logistic regression   # Use in Classsification Problem
        #  Logistic regression  Output as probabilities : [P<0.5 =0, p>0.5 =1]
        #  US_Voting_Filtered_Data

        import pandas as pd
        df=pd.read_csv('data_files/US_Voting_Filtered_Data_.csv')  
        X = df.drop('party', axis=1).values
        y = df['party'].values

        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.4,random_state=42,stratify=y)

        from sklearn.linear_model import LogisticRegression
        logreg=LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        print("score :",logreg.score(X_test,y_test))

    def ROC_():
            # 29    ->  # 28 Continuation 
        # ROC :Reciver Operating Charasterstic curve 
        #  Logistic regression Probabilities 
        # Voting Data
        import pandas as pd


        df=pd.read_csv('data_files/US_Voting_Filtered_Data_.csv')  
        X = df.drop('party', axis=1).values
        y = df['party'].values

        y_test=pd.DataFrame(y_test)
        y_test.replace(('republican', 'democrat'), (1, 0), inplace=True)   

        y_pred_prob= logreg.predict_proba(X_test)[:,1]     #  [P<0.5 =0, p>0.5 =1] -> Out put two Columns 

        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds= roc_curve(y_test, y_pred_prob)

        import matplotlib.pyplot as plt
        plt.plot([0,1],[0,1],'k--')
        plt.plot(fpr, tpr,label='Logistic Regression')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Logistic Regression ROC Curve')
        plt.show()


    def LogisticRegression_Classif_ConfMat_():
            # 30
        # Logistic REgression -> for classification Probelms
        # Diabatic Data  
        # # Confusion Matrix
        # # Classification_report



        import pandas as pd
        df=pd.read_csv('data_files/Diabetes_Data.csv')  
        df.insulin.replace((0), (155.54), inplace=True)
        X = df.drop('diabetes', axis=1).values
        y = df['diabetes'].values

        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.4,random_state=42)


        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression()
        logreg.fit(X_train,y_train)
        predict=logreg.predict(X_test)

        #Confusion Matrix
        from sklearn.metrics import confusion_matrix
        print("Confusion Matrix :\n[TP, FN]\n[FP, TN]\n",confusion_matrix(y_test,predict))  
        ## [TP, FN]  [0=0, 0=1]
        ## [FP, TN]  [1=0, 1=1]

        #Classification Report
        from sklearn.metrics import classification_report
        print("\nprecision    :For True: Do not say as False")
        print("recall       :For False: Say as False\n")
        print(classification_report(y_test, predict))

        # Accuracy Score :: not always an informative metric
        print("Score: ",logreg.score(X_test,y_test))
    def AUC_():
        # 32
        # roc_auc_score
        # AUC:Area Under Curve.,  ROC:Reciver operating Characterstic Curve.
        # Diabetes data : Logistic REgression
        import pandas as pd
        df=pd.read_csv('data_files/Diabetes_Data.csv')  
        df.insulin.replace((0), (155.54), inplace=True)
        X = df.drop('diabetes', axis=1).values
        y = df['diabetes'].values

        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.4,random_state=42)

        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression()
        logreg.fit(X_train,y_train)
        predict=logreg.predict(X_test)
        y_pred_prob = logreg.predict_proba(X_test)[:,1]


        from sklearn.metrics import roc_auc_score
        RocAucScore=roc_auc_score(y_test, y_pred_prob)
        print("roc_auc_score:",RocAucScore)



    def AUC_CV_():
        # 33  Contineoution of 32 
        # AUC  Cross Validation
        import pandas as pd
        df=pd.read_csv('data_files/Diabetes_Data.csv')  
        df.insulin.replace((0), (155.54), inplace=True)
        X = df.drop('diabetes', axis=1).values
        y = df['diabetes'].values

        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(logreg, X,y,cv=5,scoring='roc_auc')
        print("AUC scores computed using 5-fold cross-validation: ",cv_scores)

    def HyperParameterTuning_GridSearch_Knn():
        # 34
        # HyperParameter Tuning.  
        # HyperParameter like Alpha in regression, N in Knn.
        # HyperParameter Must Define Before Fitting the Model.

        # Grid Search                  # GridSearchCV can be computationally expensive.
        # KNeighborsClassifier

        import pandas as pd
        df=pd.read_csv('data_files/Diabetes_Data.csv')  
        df.insulin.replace((0), (155.54), inplace=True)
        X = df.drop('diabetes', axis=1).values
        y = df['diabetes'].values


        import numpy as np
        param_grid ={'n_neighbors':np.arange(1,50)}   # Specificing Grid Hyperparameters   #n_neighbors/Alpha


        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier() 

        from sklearn.model_selection import GridSearchCV 
        knn_cv= GridSearchCV(knn, param_grid , cv=5) # (algo, grid, nooffolds)

        knn_cv.fit(X,y)
        best_parameters=knn_cv.best_params_
        print("best_parameters: ",best_parameters)
        best_score=knn_cv.best_score_
        print("best_score: ",best_score)

    def HyperParameterTuning_GridSearch_LogisticRegr():
        # 35
        # HyperParameter Tuning.    "C"
        # logistic regression       "C"
        # Diabetes Data

        import pandas as pd
        df=pd.read_csv('data_files/Diabetes_Data.csv')  
        df.insulin.replace((0), (155.54), inplace=True)
        X = df.drop('diabetes', axis=1).values
        y = df['diabetes'].values



        c_space = np.logspace(-5, 8, 15)
        param_grid = {'C': c_space}       # HyperParameter "C" for logistic regression classifier

        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression()

        from sklearn.model_selection import GridSearchCV
        logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

        logreg_cv.fit(X, y)

        print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))  # best_params_
        print("Best score is {}".format(logreg_cv.best_score_))  # best_score_


    def HyperParameterTuning_RandSearch_DecisionTreeClassifier():
        # 36
        # HyperParameter Tuning. 
        # RandomizedSearchCV
        # RandomizedSearchCV can be computationally expensive than GridSearchCV   ????
        # DecisionTreeClassifier

        import pandas as pd
        df=pd.read_csv('data_files/Diabetes_Data.csv')  
        df.insulin.replace((0), (155.54), inplace=True)
        X = df.drop('diabetes', axis=1).values
        y = df['diabetes'].values


        from scipy.stats import randint                   # randam forest
        param_dist = {"max_depth": [3, None],             # HyperParameters !
                      "max_features": randint(1, 9),
                      "min_samples_leaf": randint(1, 9),
                      "criterion": ["gini", "entropy"]}


        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier()

        from sklearn.model_selection import RandomizedSearchCV
        tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
        tree_cv.fit(X,y)
        # I guess labels error check how to get best index and best score.
        print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_index_))
        print("Best score is {}".format(tree_cv.best_score_))



    def DecisionTreeClassifier_(): 
        # 36+
        # HyperParameter Tuning. 
        # RandomizedSearchCV
        # RandomizedSearchCV can be computationally expensive than GridSearchCV   ????
        # DecisionTreeClassifier

        import pandas as pd
        df=pd.read_csv('data_files/Diabetes_Data.csv')  
        df.insulin.replace((0), (155.54), inplace=True)
        X = df.drop('diabetes', axis=1).values
        y = df['diabetes'].values



        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier()
        tree.fit(X,y)

        print("Best score is {}".format(tree.score(X,y)))

    def HyperParameterTuning_GridSearch_ElasticNetRegularization_():  
        # 38
        # In elastic net regularization  = linear combination of the L1 [Lasso] and L2[Ridge]  penalties = a∗L1+b∗L2
        # Lasso used the L1 penalty to regularize
        # ridge used the L2 penalty

        # scikit-learn, this term is represented by the 'l1_ratio' parameter: 
        # An 'l1_ratio' of 1 corresponds to an L1 penalty,
        # and anything lower is a combination of L1 and L2.

        # GridSearchCV 
        # to tune l1_ratio
        # Gapminder data.
        # ElasticNet regressor 


        from sklearn.metrics import mean_squared_error

        from sklearn.model_selection import train_test_split

        import pandas as pd
        df = pd.read_csv('data_files/Gapminder_Region_data.csv')
        y = df.life.values.reshape(-1,1)
        X= df.drop(["life","Region"],axis=1).values


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

        l1_space = np.linspace(0, 1, 30) 
        param_grid = {'l1_ratio': l1_space}  # Hyperparametes

        from sklearn.linear_model import ElasticNet     
        elastic_net = ElasticNet()       # ElasticNet regressor = Lasso + Ridge

        from sklearn.model_selection import GridSearchCV
        gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)


        gm_cv.fit(X_train, y_train)
        y_pred = gm_cv.predict(X_test)

        r2 = gm_cv.score(X_test, y_test)          # Score
        mse = mean_squared_error(y_test, y_pred)  # Mean Square Error

        print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
        print("Tuned ElasticNet R squared: {}".format(r2))
        print("Tuned ElasticNet MSE: {}".format(mse))

    def ElasticNetRegularization_():
        # 38
        # In elastic net regularization  = linear combination of the L1 [Lasso] and L2[Ridge]  penalties = a∗L1+b∗L2
        # Lasso used the L1 penalty to regularize
        # ridge used the L2 penalty

        # scikit-learn, this term is represented by the 'l1_ratio' parameter: 
        # An 'l1_ratio' of 1 corresponds to an L1 penalty,
        # and anything lower is a combination of L1 and L2.



        from sklearn.metrics import mean_squared_error

        from sklearn.model_selection import train_test_split

        import pandas as pd
        df = pd.read_csv('data_files/Gapminder_Region_data.csv')
        y = df.life.values.reshape(-1,1)
        X= df.drop(["life","Region"],axis=1).values


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)


        from sklearn.linear_model import ElasticNet     
        elastic_net = ElasticNet()       # ElasticNet regressor = Lasso + Ridge



        elastic_net.fit(X_train, y_train)
        y_pred = elastic_net.predict(X_test)

        r2 = elastic_net.score(X_test, y_test)          # Score
        print(r2)

    def Preprocessing_Char2Num_():
            # 39
        # Unit 4 
        # PreProcessing Data
        # get_dummies
        # Ex red/Blue -> 1/2
        #  -> Scikit Learn: OneHotEncoder()
        #  -> pandas:get_dummies()
        import pandas as pd
        df = pd.read_csv('data_files/Automobile_MilesPerGallon_dataset.csv')


        df_origin = pd.get_dummies(df)   # # get_dummies
        print(df.head())
        print(df_origin.head())

        df_origin=df_origin.drop('origin_Europe',axis=1)  # Dropped origin_Europe
        # df_region = pd.get_dummies(df, drop_first=True) # Direct drop 1th dummy
        print(df_origin.head())


    def Box_plot_():
        # 40
        # Box Plot

        import pandas as pd
        df = pd.read_csv('data_files/Gapminder_Region_data.csv')

        df.boxplot('life', 'Region', rot=60)

        import matplotlib.pyplot as plt
        print("life vs Region PLOT")
        plt.show()

    def Preprocessing_DummyDrop_():
        # 41
        # Create and drop dummy variable

        import pandas as pd
        df = pd.read_csv('data_files/Gapminder_Region_data.csv')

        df_region = pd.get_dummies(df)
        # print(df_region.head())

        print(df_region.columns)

        df_region = pd.get_dummies(df, drop_first=True)
        print("\nRegion_America Dropped\n")
        print(df_region.columns)

    def RidgeRegression_DropDummy_():
        # 42
        # RidgeRegression  + Create and drop dummy variable + cv on Gapminder DAta
        import pandas as pd
        df = pd.read_csv('data_files/Gapminder_Region_data.csv')
        y = df.life.values.reshape(-1,1)

        df_region = pd.get_dummies(df, drop_first=True)   # Dummies Drop implement

        X= df_region.drop(["life"],axis=1).values


        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=0.5, normalize=True)

        from sklearn.model_selection import cross_val_score
        ridge_cv = cross_val_score(ridge, X, y, cv=5)

        print(ridge_cv)

    def Preprocessing_dropNan_():
            # 44 contineoution of 43
        # Method 1 Drop
        # dropping Nan
        # drop  cause huge loss of data
        print("Before drop:" ,df.shape)
        df= df.dropna()  #Drop Nan
        print("After drop : ",df.shape)
    def Preprocessing_Replace():
        # 43
        # Handling Missing Data
        # PIMA indian Diabetes data set
        # Missing data as  insulin 0 : triceps 0 : bmi 0
        # Missing as : 0 ? nan '9999'
        import pandas as pd
        import numpy as np

        df= pd.read_csv('data_files/Diabetes_Data.csv')
        # print(df.info())
        # print(df.head()) 

        # 0 -> nan
        # df[df == '?'] = np.nan   # One Step Replacement
        # df[df == 0] = np.nan
        df.insulin.replace(0,np.nan,inplace=True)
        df.triceps.replace(0,np.nan,inplace=True) 
        df.bmi.replace(0,np.nan, inplace=True)



    def Preprocessing_Imputer():
        # 45  
        # Method 2 Mean/Median/.. replacement -> to Nan
        # Using Imputer
        # Imputers are known as transformer
        # No loss of data

        import pandas as pd
        import numpy as np

        df= pd.read_csv('data_files/Diabetes_Data.csv')
        # print(df.info())
        # print(df.head()) 

        # 0 -> nan
        df.insulin.replace(0,np.nan,inplace=True)
        df.triceps.replace(0,np.nan,inplace=True) 
        df.bmi.replace(0,np.nan, inplace=True)


        print(df.head(1))
        from sklearn.preprocessing import Imputer
        imp = Imputer(missing_values='NaN', strategy='mean',axis=0)   # Initialize Instance   # [axis =0 Column axis =1 Rows]
        imp.fit(df)
        df=imp.transform(df)
        print("Mean/Median/Mode :",df[0])

    def SVM_Class_ImpPip_():
        # 48 
        # SVM: Support Vector Machine
        # Imputer 
        # Pipeline
        # classification_report
        # US_Voting_Data
        import pandas as pd
        df=pd.read_csv("data_files/US_Voting_Data.csv")
        df.replace(('y', 'n'), (1, 0), inplace=True) # df_region = pd.get_dummies(df)
        df[df == '?'] = np.nan

        X = df.drop('party', axis=1) 
        y = df['party'] 

        from sklearn.preprocessing import Imputer
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline

        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0) # NOT MEAN WE USE MOST FREQUENT ONE TO REPLACE

        clf = SVC()                      # SVC classifier: clf

        steps = [('imputation', imp), ('SVM', clf)]

        pipeline = Pipeline(steps)       # Initialize Stepss         Imputer-> SVC classifier

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        from sklearn.metrics import classification_report   # // yes No Accuracy 
        print(classification_report(y_test, y_pred))

    def Preprocessing_Cent_Scal_():
        # 49
        # Centering and Scaline !
        # Data Relatively high difference influence the model 
        # Ex data_1 :0.001 , data_2:1000000        Need to normalize both the columns
        # we want Features to be on similar scale by # Centering and Scaline !
        df=pd.read_csv("White_wine_data.csv")
        print("Check Mean for relative value difference\n")
        print(df.describe())

    def Preprocessing_Scal_():
        # 50 
        # Scale
        from sklearn.preprocessing import scale
        import numpy as np
        df=pd.read_csv("White_wine_data.csv")
        X_scaled=scale(df.iloc[0])
        print(" \nUnscaled: \n",np.mean(df.iloc[0] ),np.std(df.iloc[0] ))
        print(" \nscaled:\n ",np.mean(X_scaled ),np.std(X_scaled ))




    def KNeighborsClassifier_ScaledUnscaled_():
        #51
        #StandardScaler
        #Pipeline
        #Knn
        #################################################
        import pandas as pd
        df= pd.read_csv('data_files/Diabetes_Data.csv')
        # 0 -> nan
        df.insulin.replace(0,np.nan,inplace=True)
        df.triceps.replace(0,np.nan,inplace=True) 
        df.bmi.replace(0,np.nan, inplace=True)
        from sklearn.preprocessing import Imputer
        imp = Imputer(missing_values='NaN', strategy='mean',axis=0)   # Initialize Instance   # [axis =0 Column axis =1 Rows]
        imp.fit(df)
        X = df.drop('diabetes', axis=1) 
        X=imp.transform(df)
        y = df['diabetes'] 
        #################################################

        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        knn= KNeighborsClassifier(n_neighbors=6)



        from sklearn.pipeline import Pipeline
        steps = [('scaler',StandardScaler()),('knn', KNeighborsClassifier())]
        pipeline = Pipeline(steps)


        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=21)

        knn_scaled =pipeline.fit(X_train,y_train)
        y_pred = knn_scaled.predict(X_test)

        scaled_accu=knn_scaled.score(X_test,y_test)
        print("Scaled Accuracy : ", scaled_accu)

        scaled_accu=KNeighborsClassifier().fit(X_train,y_train).score(X_test,y_test)          # Cool every thing in one line
        print("UnScaled Accuracy : ", scaled_accu)

    def KneighorsClassifier_PipeScalCVGridsearc_():
        #52
        # CV  and pipeline

        from sklearn import datasets
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target


        from sklearn.pipeline import Pipeline
        steps = [('scaler',StandardScaler()),('knn', KNeighborsClassifier())]


        from sklearn.pipeline import Pipeline
        steps = [('scaler',StandardScaler()),('knn', KNeighborsClassifier())]
        pipeline = Pipeline(steps)

        parameters = {'knn__n_neighbors':np.arange(1,50)}


        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=21)

        from sklearn.model_selection import GridSearchCV
        cv= GridSearchCV(pipeline, param_grid=parameters) # determine knn

        cv.fit(X_train, y_train)
        y_pred = cv.predict(X_test)

        print ("Best Knn n: ",cv.best_params_)

        print("score: ",cv.score(X_test, y_test))

        from sklearn.metrics import classification_report 
        print("classification_report: ",classification_report(y_test,y_pred))

    def Preprocess_Scale_():
        # 53 
        # scale 
        # binary: In such a situation, scaling will have minimal impact
        # to scale the features
        # density = varies 0to 1
        # 'total sulfur dioxide'  varies  0 to 289
        from sklearn.preprocessing import scale
        import pandas as pd
        df= pd.read_csv('data_files/White_wine_data.csv')
        X_scaled = scale(df)
        X = df.values          # in terms of array not table format

        # Print the mean and standard deviation of the unscaled features
        print("Mean of Unscaled Features: {}".format(np.mean(X))) 
        print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

        # Print the mean and standard deviation of the scaled features
        print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
        print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

    def KNeighborsClassifier_PipeScalCV_():
        # 54
        # Centering and scaling in a pipeline
        # k-NN classifier 
        # with Scaling and with out scaling Accuracy

        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        df= pd.read_csv('data_files/White_wine_data.csv') # taken only 1000 values
        X=df.values[0:1000]
        y=np.array([False, False, False, False, False, False, False, False, False,       False,  True,  True,  True, False,  True, False, False, False,       False,  True, False, False, False,  True, False, False, False,       False, False, False, False, False, False, False,  True,  True,        True, False,  True,  True, False, False, False, False, False,       False,  True,  True, False,  True, False, False, False, False,       False, False, False, False, False, False, False, False,  True,       False, False,  True, False,  True, False,  True, False,  True,        True, False, False,  True, False, False,  True,  True, False,       False,  True, False,  True, False, False, False,  True, False,       False,  True, False, False, False, False, False, False,  True,       False,  True,  True,  True,  True,  True, False,  True, False,       False,  True, False,  True,  True,  True,  True,  True, False,       False,  True,  True,  True,  True,  True, False, False, False,        True, False, False, False,  True, False,  True,  True,  True,        True, False,  True, False, False,  True,  True, False, False,       False, False, False,  True, False, False, False, False, False,        True, False, False, False, False, False, False, False,  True,        True, False,  True,  True, False, False,  True,  True, False,       False,  True, False,  True, False,  True,  True,  True, False,       False,  True,  True, False,  True,  True, False,  True, False,        True, False,  True, False,  True,  True, False,  True,  True,        True,  True,  True,  True,  True, False,  True,  True,  True,        True,  True, False,  True, False,  True, False, False,  True,        True,  True,  True,  True,  True, False, False, False, False,        True, False, False, False,  True,  True, False, False, False,       False, False, False, False, False, False,  True,  True, False,       False,  True, False, False, False, False,  True,  True,  True,        True,  True, False, False, False, False, False,  True, False,        True,  True, False, False,  True, False,  True, False, False,       False,  True,  True,  True,  True, False, False,  True,  True,       False, False, False,  True,  True,  True,  True, False, False,       False, False, False, False,  True, False,  True, False,  True,       False, False, False, False, False, False, False, False, False,        True, False, False, False, False, False, False, False,  True,       False, False,  True, False, False, False,  True, False, False,        True,  True, False, False, False,  True, False,  True, False,        True,  True, False, False, False,  True, False, False, False,       False,  True, False, False, False, False, False,  True, False,       False, False, False, False, False, False, False, False, False,       False,  True, False, False, False, False, False, False, False,        True, False, False,  True, False, False, False, False, False,       False, False, False, False, False, False, False, False, False,        True, False, False, False,  True, False, False,  True,  True,        True, False,  True, False, False,  True,  True,  True, False,        True, False,  True, False,  True, False, False,  True,  True,       False, False, False,  True, False, False, False, False, False,       False, False, False, False,  True,  True,  True,  True,  True,       False,  True, False, False,  True, False, False,  True, False,       False, False, False, False,  True,  True, False, False, False,        True,  True, False, False, False, False, False,  True, False,        True,  True,  True,  True, False,  True,  True, False, False,        True,  True, False,  True, False, False, False,  True, False,       False, False, False,  True, False,  True,  True,  True, False,       False, False, False, False, False, False, False, False, False,       False,  True, False,  True,  True, False, False, False,  True,       False, False,  True, False, False, False, False, False, False,       False, False, False,  True, False, False,  True,  True,  True,       False, False,  True, False,  True, False, False, False, False,        True, False, False, False,  True,  True, False,  True, False,        True,  True, False, False, False, False, False, False, False,        True, False, False, False, False, False, False,  True, False,        True, False, False,  True, False, False,  True, False, False,        True, False, False,  True, False,  True, False, False, False,       False, False, False, False,  True,  True, False, False, False,       False, False, False, False, False,  True, False,  True,  True,        True, False,  True, False, False, False, False, False,  True,        True, False, False,  True,  True,  True, False, False, False,        True,  True,  True,  True, False, False, False, False,  True,        True, False,  True,  True, False,  True, False, False, False,        True,  True, False,  True, False, False, False,  True,  True,        True, False,  True, False,  True,  True,  True,  True, False,        True, False, False, False, False, False, False, False, False,        True,  True,  True,  True, False,  True,  True, False,  True,       False, False, False,  True, False, False, False, False,  True,       False, False, False, False, False,  True,  True, False,  True,        True,  True, False,  True, False, False,  True, False,  True,        True, False,  True, False,  True,  True,  True,  True, False,        True, False, False,  True,  True, False, False,  True,  True,       False, False,  True, False, False, False,  True, False, False,        True,  True, False, False, False,  True, False,  True,  True,        True, False, False, False, False,  True, False, False,  True,       False, False,  True, False, False,  True,  True, False, False,       False, False, False, False, False, False, False,  True, False,        True, False, False, False, False, False,  True, False, False,       False,  True, False, False, False, False, False, False,  True,       False, False, False,  True, False, False,  True, False, False,       False,  True, False, False, False, False, False, False, False,        True, False,  True,  True, False, False, False, False, False,        True,  True, False, False, False,  True, False, False, False,        True, False, False, False, False,  True,  True,  True,  True,        True, False, False,  True, False,  True, False, False, False,       False, False, False,  True, False, False, False, False, False,       False, False, False, False, False, False, False, False, False,        True,  True, False, False, False,  True, False,  True, False,       False,  True,  True, False,  True, False, False, False, False,        True, False, False, False, False, False,  True,  True, False,        True,  True, False, False, False, False, False, False, False,       False, False, False, False, False, False, False, False, False,       False,  True, False, False, False, False, False, False, False,       False, False,  True, False, False,  True,  True, False,  True,        True,  True,  True,  True,  True,  True,  True, False, False,       False, False, False, False, False, False, False,  True,  True,       False,  True,  True, False,  True, False,  True,  True,  True,        True,  True, False, False,  True, False, False, False, False,       False,  True,  True,  True,  True,  True, False, False,  True,       False,  True,  True, False, False, False, False, False, False,       False, False,  True, False, False, False, False, False, False,       False, False, False,  True, False,  True, False,  True, False,       False, False, False,  True, False, False, False,  True, False,       False,  True,  True,  True, False, False,  True, False, False, False], dtype=bool)

        from sklearn.pipeline import Pipeline
        steps = [('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())]
        pipeline = Pipeline(steps)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        knn_scaled = pipeline.fit(X_train, y_train)
        knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)


        print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
        print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

    def SVM_PipeScalCVGridsearc_():
        # 55
        # Building all to gether
                  # For Classification
        #SVM classifier
        # C and gamma. C controls the regularization strength
        # gamma controls the kernel coefficient
        # Pipline -> Scaler -> SVM -> GridSearchCv
        #White_wine_data set


        import pandas as pd
        df= pd.read_csv('data_files/White_wine_data.csv') # taken only 1000 values
        X=df.values[0:1000]
        y=np.array([False, False, False, False, False, False, False, False, False,       False,  True,  True,  True, False,  True, False, False, False,       False,  True, False, False, False,  True, False, False, False,       False, False, False, False, False, False, False,  True,  True,        True, False,  True,  True, False, False, False, False, False,       False,  True,  True, False,  True, False, False, False, False,       False, False, False, False, False, False, False, False,  True,       False, False,  True, False,  True, False,  True, False,  True,        True, False, False,  True, False, False,  True,  True, False,       False,  True, False,  True, False, False, False,  True, False,       False,  True, False, False, False, False, False, False,  True,       False,  True,  True,  True,  True,  True, False,  True, False,       False,  True, False,  True,  True,  True,  True,  True, False,       False,  True,  True,  True,  True,  True, False, False, False,        True, False, False, False,  True, False,  True,  True,  True,        True, False,  True, False, False,  True,  True, False, False,       False, False, False,  True, False, False, False, False, False,        True, False, False, False, False, False, False, False,  True,        True, False,  True,  True, False, False,  True,  True, False,       False,  True, False,  True, False,  True,  True,  True, False,       False,  True,  True, False,  True,  True, False,  True, False,        True, False,  True, False,  True,  True, False,  True,  True,        True,  True,  True,  True,  True, False,  True,  True,  True,        True,  True, False,  True, False,  True, False, False,  True,        True,  True,  True,  True,  True, False, False, False, False,        True, False, False, False,  True,  True, False, False, False,       False, False, False, False, False, False,  True,  True, False,       False,  True, False, False, False, False,  True,  True,  True,        True,  True, False, False, False, False, False,  True, False,        True,  True, False, False,  True, False,  True, False, False,       False,  True,  True,  True,  True, False, False,  True,  True,       False, False, False,  True,  True,  True,  True, False, False,       False, False, False, False,  True, False,  True, False,  True,       False, False, False, False, False, False, False, False, False,        True, False, False, False, False, False, False, False,  True,       False, False,  True, False, False, False,  True, False, False,        True,  True, False, False, False,  True, False,  True, False,        True,  True, False, False, False,  True, False, False, False,       False,  True, False, False, False, False, False,  True, False,       False, False, False, False, False, False, False, False, False,       False,  True, False, False, False, False, False, False, False,        True, False, False,  True, False, False, False, False, False,       False, False, False, False, False, False, False, False, False,        True, False, False, False,  True, False, False,  True,  True,        True, False,  True, False, False,  True,  True,  True, False,        True, False,  True, False,  True, False, False,  True,  True,       False, False, False,  True, False, False, False, False, False,       False, False, False, False,  True,  True,  True,  True,  True,       False,  True, False, False,  True, False, False,  True, False,       False, False, False, False,  True,  True, False, False, False,        True,  True, False, False, False, False, False,  True, False,        True,  True,  True,  True, False,  True,  True, False, False,        True,  True, False,  True, False, False, False,  True, False,       False, False, False,  True, False,  True,  True,  True, False,       False, False, False, False, False, False, False, False, False,       False,  True, False,  True,  True, False, False, False,  True,       False, False,  True, False, False, False, False, False, False,       False, False, False,  True, False, False,  True,  True,  True,       False, False,  True, False,  True, False, False, False, False,        True, False, False, False,  True,  True, False,  True, False,        True,  True, False, False, False, False, False, False, False,        True, False, False, False, False, False, False,  True, False,        True, False, False,  True, False, False,  True, False, False,        True, False, False,  True, False,  True, False, False, False,       False, False, False, False,  True,  True, False, False, False,       False, False, False, False, False,  True, False,  True,  True,        True, False,  True, False, False, False, False, False,  True,        True, False, False,  True,  True,  True, False, False, False,        True,  True,  True,  True, False, False, False, False,  True,        True, False,  True,  True, False,  True, False, False, False,        True,  True, False,  True, False, False, False,  True,  True,        True, False,  True, False,  True,  True,  True,  True, False,        True, False, False, False, False, False, False, False, False,        True,  True,  True,  True, False,  True,  True, False,  True,       False, False, False,  True, False, False, False, False,  True,       False, False, False, False, False,  True,  True, False,  True,        True,  True, False,  True, False, False,  True, False,  True,        True, False,  True, False,  True,  True,  True,  True, False,        True, False, False,  True,  True, False, False,  True,  True,       False, False,  True, False, False, False,  True, False, False,        True,  True, False, False, False,  True, False,  True,  True,        True, False, False, False, False,  True, False, False,  True,       False, False,  True, False, False,  True,  True, False, False,       False, False, False, False, False, False, False,  True, False,        True, False, False, False, False, False,  True, False, False,       False,  True, False, False, False, False, False, False,  True,       False, False, False,  True, False, False,  True, False, False,       False,  True, False, False, False, False, False, False, False,        True, False,  True,  True, False, False, False, False, False,        True,  True, False, False, False,  True, False, False, False,        True, False, False, False, False,  True,  True,  True,  True,        True, False, False,  True, False,  True, False, False, False,       False, False, False,  True, False, False, False, False, False,       False, False, False, False, False, False, False, False, False,        True,  True, False, False, False,  True, False,  True, False,       False,  True,  True, False,  True, False, False, False, False,        True, False, False, False, False, False,  True,  True, False,        True,  True, False, False, False, False, False, False, False,       False, False, False, False, False, False, False, False, False,       False,  True, False, False, False, False, False, False, False,       False, False,  True, False, False,  True,  True, False,  True,        True,  True,  True,  True,  True,  True,  True, False, False,       False, False, False, False, False, False, False,  True,  True,       False,  True,  True, False,  True, False,  True,  True,  True,        True,  True, False, False,  True, False, False, False, False,       False,  True,  True,  True,  True,  True, False, False,  True,       False,  True,  True, False, False, False, False, False, False,       False, False,  True, False, False, False, False, False, False,       False, False, False,  True, False,  True, False,  True, False,       False, False, False,  True, False, False, False,  True, False,       False,  True,  True,  True, False, False,  True, False, False, False], dtype=bool)

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        steps = [('scaler', StandardScaler()), ('SVM', SVC())] # Pipeline
        pipeline = Pipeline(steps)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

        from sklearn.model_selection import GridSearchCV
        parameters = {'SVM__C':[1, 10, 100],'SVM__gamma':[0.1, 0.01]} # hyperparameter space
        cv = GridSearchCV(pipeline, parameters)

        cv.fit(X_train, y_train)
        y_pred = cv.predict(X_test)


        print("Accuracy: {}".format(cv.score(X_test, y_test)))
        from sklearn.metrics import classification_report 
        print(classification_report(y_test, y_pred))
        print("Tuned Model Parameters: {}".format(cv.best_params_))



    def ElasticNet_PipeImputerScaleGridsearch_():
        # 56
        # Building all to gether
                  # For Regression
        # Gapminder dataset
        #ElasticNet 
        # pipeline  -> meanForMissingData -> Scaline -> ElasticNetRegression-> GridSearchCV



        ###########################################################
        import pandas as pd
        df= pd.read_csv('data_files/Gapminder_Region_data.csv')
        # df = pd.get_dummies(df, drop_first=True)
        y = df.life.values# reshape(-1,1)
        X= df.drop(["life","Region"],axis=1).values
        ################################
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import Imputer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import ElasticNet   
        steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
                 ('scaler', StandardScaler()),
                 ('elasticnet', ElasticNet())]
        pipeline = Pipeline(steps)


        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


        from sklearn.model_selection import GridSearchCV
        parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}  # Hyperparameter space
        gm_cv = GridSearchCV(pipeline, parameters)
        gm_cv.fit(X_train, y_train)

        r2 = gm_cv.score(X_test, y_test)
        print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
        print("Tuned ElasticNet R squared Score: {}".format(r2))





    
    
