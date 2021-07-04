library(ggplot2)
library(ggthemes)
library(hrbrthemes)
library(viridis)

# 1 ---- [13, x, 3], 250
nodes <- as.factor(c(1, 5, 10, 25, 50))
accuracy1 <- c(0.6235955056179775, 0.9606741573033708, 0.9831460674157303, 0.9775280898876404, 0.9831460674157303)

df1 <- data.frame(nodes, accuracy1)

ggplot(data = df1, aes(x = nodes, y = accuracy1)) + 
    geom_bar(stat='identity', fill = nodes)+
    labs(title="Accuracy of NN Using Different Number of Nodes\n in 1 Hidden Layer with 250 Epochs") + 
    xlab("Nodes") +
    ylab("Accuracy") +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme_economist() +
    coord_cartesian(ylim=c(0.6,1.05))
 
# 2 ---- [13, [x] * {0-9}, 3], 100
hidden_layers <- seq(0,9,1)
accuracy2 <- c(0.10674157303370786
              ,0.9775280898876404
              ,0.9325842696629213
              ,0.5280898876404494
              ,0.2696629213483146
              ,0.2696629213483146
              ,0.2696629213483146
              ,0.0
              ,0.2696629213483146
              ,0.0)

df2 <- data.frame(hidden_layers, accuracy2)


ggplot(data = df2, aes(x = hidden_layers, y = accuracy2))+
    geom_line(size=1) +
    geom_text(aes(label=round(accuracy2, 2)), position=position_dodge(width=0.8), vjust=-0.7, hjust=0, size=4.5) +
    geom_point(color = "blue", size = 2) +
    labs(title="Accuracy as Number of Hidden Layers Change (10 Nodes per Layer, 100 Epochs)") + 
    xlab("Number of Hidden Layer") +
    ylab("Accuracy") +
    theme(plot.title = element_text(hjust = 0.5)) +
    scale_x_continuous(breaks = hidden_layers) +
    theme_stata() 



# 3 ---- [13, factor(24), 3], 100
layer_distribution <- c("24", "12-12", "8-8-8", "6-6-6-6", "4-4-4-4-4-4", "3-3-3-3-3-3-3-3")
accuracy3 <- c(0.9719101123595506
              ,0.898876404494382
              ,0.5056179775280899
              ,0.5955056179775281
              ,0.2696629213483146
              ,0.2696629213483146)

df3 <- data.frame(layer_distribution, accuracy3)


ggplot(data=df3, aes(x= layer_distribution, y= accuracy3)) +
    geom_bar(stat='identity', fill = layer_distribution) +
    labs(title="Accuracy vs Different Distributions of Layers with 24 Total Nodes (100 Epochs)") + 
    xlab("Layer Distributions") +
    ylab("Accuracy") +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme_ipsum() +
    theme_light() +
    coord_flip()


# 4 ---- [13, 12, 12, 3], 10 ~ 250
epochs <- seq(10, 250, 30)
accuracy4 <- c(0.8033707865168539
              ,0.797752808988764
              ,0.9719101123595506
              ,0.9438202247191011
              ,0.9719101123595506
              ,0.9775280898876404
              ,0.9831460674157303
              ,0.9887640449438202
              ,0.9943820224719101)

df4 <- data.frame(epochs, accuracy4)

ggplot(data = df4, aes(x = epochs, y = accuracy4))+
    geom_line(size=1) +
    geom_text(aes(label=round(accuracy4, 2)), position=position_dodge(width=0.8), vjust=-0.7, hjust=1.5, size=4) +
    geom_point(color = "blue", size = 2) +
    labs(title="Accuracy vs Epochs (2 Hidden Layer, 12 Nodes Each)") + 
    xlab("Epoch") +
    ylab("Accuracy") +
    theme(plot.title = element_text(hjust = 0.5)) +
    scale_x_continuous(breaks = epochs) +
    theme_ipsum()



# 5 ----  ([13] + i + [3], 100)
architecture <- c("20-20-20", "10-20-30", "10-30-20", "20-10-30", "20-30-10", "30-20-10", "30-10-20")
accuracy5 <- c(0.5898876404494382
               ,0.6460674157303371
               ,0.6741573033707865
               ,0.5224719101123596
               ,0.6292134831460674
               ,0.5786516853932584
               ,0.8651685393258427)

df5 <- data.frame(architecture, accuracy5)



ggplot(data = df5, aes(x = architecture, y = accuracy5)) +
    geom_bar(stat = "identity", fill = architecture) +
    labs(title="Accuracy vs Different Hidden Layer Structures (60 Nodes Total, 100 Epochs)") + 
    xlab("Architecture") +
    ylab("Accuracy") +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme_light()
