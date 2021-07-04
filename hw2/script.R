library(tidyverse)


d <- read.csv("C:\\Users\\guozh\\Desktop\\CS-375\\HW2\\data.csv", row.names = 1)
d <- d[-6,]
d <- d[, -6]


x <- data.frame("Black Wins" = sum(d > 0), "White Wins" = sum(d < 0))

x <- data.frame(
    Winner = c("Black Wins", "White Wins"),
    Count = c(sum(d > 0), sum(d < 0))
)
bp<- ggplot(x, aes(x="", y=Count, fill=Winner))+
    geom_bar(width = 1, stat = "identity")


pie <- bp + coord_polar("y", start=0)
pie + scale_fill_manual(values=c("Black","White")) + theme_minimal()


