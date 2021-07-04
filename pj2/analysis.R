library(tidyverse)
library(ggplot2)
library(dplyr)
library(patchwork) # To display 2 charts together
library(hrbrthemes)

algorithm <- as.factor(c("Minimax", "Minimax","Minimax", "ABPrunning", "ABPrunning", "ABPrunning"))
opponent <- as.factor(c("ABPrunning", "STMax", "OMMin", "Minimax", "STMax", "OMMin"))
runtime <- c(25.74423360824585, 198.65266227722168, 33.34000778198242, 7.959728240966797, 30.138489484786987, 10.291491985321045)

df1 <- data.frame(algorithm, opponent, runtime)

ggplot(df1, aes(x = opponent, y = runtime, fill = algorithm)) + 
    geom_bar(stat="identity", position = "dodge") + 
    scale_fill_brewer(palette = "Set1") + 
    ggtitle("Time to Finish Game for Alpha-Beta Pruning vs Minimax \n
            (Both on Black, Using simple_heuristics, With Search Depth of 4)") +
    xlab("Opponent algorithm") + ylab("Runtime (sec)") + theme(plot.title = element_text(hjust = 0.5))



Heuristic <- c("Everything", "w/out Counts", "w/out Corners", "w/out Stability", "w/out Stability + Mobility")
result <- c(48, 58, -4, 34, 54)
runtime <- c(129.29250359535217, 159.75660228729248, 113.68354868888855, 92.34472060203552, 96.2622663974762)

df2 <- data.frame(Heuristic, result, runtime)
coeff <- 10


ggplot(data = df2, aes(x=Heuristic)) +
    geom_bar(aes(y=result),  stat='identity', width = 0.5, size = 0.6, fill = "blue") + 
    geom_line(aes(y = runtime, group = 1), size=2, color = "red") +
    scale_y_continuous(name = "Piece count (W - B)", sec.axis = sec_axis(~., name="Runtime in (sec)")) + 
    theme_ipsum() +  
    theme(
        axis.title.y = element_text(color = "blue", size=15),
        axis.title.y.right = element_text(color = "red", size=15)
    ) + 
    labs(title="Compare Factors in best_heuristic Against static_heuristic",
                                          subtitle = "with Alpha-Beta Pruning at depth of 5")
    + theme(plot.title = element_text(hjust = 0.5))


player <-  c("T1", "T2", "T1", "T2")
side <-  c("Black", "Black", "White", "White")
result <-  c(-28, -40, 26, -19)

df3 <- data.frame(player, side, result)

ggplot(df3, aes(x = side, y = result, fill = player)) + 
    geom_bar(stat="identity", position = "dodge") + 
    ggtitle("T1 with d=4 compared to T2 with d=5, against ABP d=4 and advanced_heuristic") +
    xlab("Color of the Player") + ylab("Result in Piece Differences") + theme(plot.title = element_text(hjust = 0.5))
