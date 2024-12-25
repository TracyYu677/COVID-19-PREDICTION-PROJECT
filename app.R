# library(shiny)
library(ggplot2)
library(plotly)

# Load in data 
xgboost <- readRDS("xgboost_output.rds")
xgboost100k <- readRDS("xgboost_100k_output.rds")
covid <- readRDS("clean_covid_data.rds")

# Define UI for app ----
ui <- navbarPage(
  
  # App title ----
  title = "Comparison of COVID case predictions by US county", 
  
  # Maps tab
  tabPanel(
    title = "Map by date",
    # App sidebar
    sidebarLayout(
      
      sidebarPanel(
        # dropdown for prediction type 
        selectInput(
          inputId = "type", 
          label = "Model",
          choices = list("XGBoost", "XGBoost100k")
        ), 
        
        # slider for day
        sliderInput(
          inputId = "dsz",
          label = "Days since Jan 21, 2020",
          min = 0,
          max = 843,
          value = 750
        ), 
        
        # numeric input option 
        numericInput(
          inputId = 'numericValue',
          label = "Day",
          min = 0,
          max = 843,
          value = 750
        ),
        
        # dropdown for prediction type
        selectInput(
          inputId = "range", 
          label = "Prediction range",
          choices = list(Day = 1, Week = 2, Month = 3, Three_months = 4, Six_months = 5)
        ), 
        width = 2
      ),
      
      # Output: main maps panel
      mainPanel(
        plotOutput("mapTest"), 
        plotOutput("mapPred")
      )
    )
   ),

  # Plots tab
  tabPanel(
    title = "Plot by FIPS",

    # sidebar9
    sidebarLayout(
      sidebarPanel(
        # Input FIPS code
        textInput(
          inputId = "fips",
          label = "County",
          value = "01003",
          placeholder = "Enter a 5-digit county FIPS code"
        ),

        # Input graph type
        selectInput(
          inputId = "type_fips",
          label = "Model",
          choices = list("None", "XGBoost", "XGBoost100k")
        ),
        width = 2
      ),

      # main
      mainPanel(
        plotlyOutput("timeSeries")
      )
    )
  )
)

# Define server logic required to draw a map ----
server <- function(input, output, session) {
  
  ######## Map Tab ########
  
  # Sync of slider and text boxes 
  v <- reactiveValues()
  #Register the current time
  v$now = Sys.time() 
  v$when = Sys.time() 
  
  observeEvent(input$numericValue, {
    v$when = Sys.time()
    req(input$numericValue) 
    if (input$numericValue != input$dsz)
    {
      updateSliderInput(
        session = session,
        inputId = 'dsz',
        value = input$numericValue
      )
    } 
  })
  
  observeEvent(input$dsz, {
    v$now = Sys.time() 
    req(input$numericValue)
    # register some input lag to prevent infinite loops 
    if (input$numericValue != input$dsz  & v$now - v$when > 0.3) 
      {
      updateNumericInput(
        session = session,
        inputId = 'numericValue',
        value = input$dsz
      )   
    }
    
  })
  
  # Output of test values for map 
  output$mapTest <- renderPlot({
    data <- pick_data(input$type)
    
    pred_map(input$dsz, 
             data, 
             "y_test", 
             paste("Change in cases over 7 days on", 
                   format(as.Date("2020-01-21") + input$dsz, "%B %d, %Y")))
  })

  # Output of predicted values 
  output$mapPred <- renderPlot({
    data <- pick_data(input$type)
    
    r <- as.numeric(input$range[[1]])
    y_data <- names(data[,4:8])
    y_shift <- c(1, 7, 30, 90, 180)
    pred_map(input$dsz, 
             data, 
             y_data[r], 
             paste("Predicted change in cases over 7 days on", 
                   format(as.Date("2020-01-21") + input$dsz, "%B %d, %Y"), 
                   "from data before", 
                   format(as.Date("2020-01-21") + input$dsz - y_shift[r], 
                          "%B %d, %Y"), 
                   "using", input$type, "model"))
  })
  
  ####### Plot Tab #######

  # plot graph
  output$timeSeries <- renderPlotly({
    time_series(input$fips, input$type_fips)
  })

}

# Function to select dataset
pick_data <- function(type) {
  if(type == "XGBoost") return(xgboost)
  else if(type == "XGBoost100k") return(xgboost100k)
}

####### Map Tab ########

# Function to extract all values from a certain date since collection, 
# and take means of values within the dates, and take the symmetric log, 
# and add county names 
extract_date <- function(dsz, data) {
  day <- data[data$days_since_zero == dsz,]
  means <- aggregate(day[,c(3:8)], by = list(day[,2]), 
                     FUN = function(x) 
                     { return(mean(x, na.rm = TRUE)) })
  names(means)[1] <- "fips"
  means[,2:7] <- symlog(means[,2:7])
  return(means)
}

# Function to provide log scale while handling negative values
symlog <- function(x, C = 0) {
  y = sign(x) * log10(1 + abs(x) / 10^C) 
  return(y)
}

# Function to plot the county map with heat colors 
pred_map <- function(dsz, data, y, title) {
  
  # pull data for date
  means <- extract_date(dsz, data)
  
  # create base map 
  usmap::plot_usmap(data = means,
                    values = y,
                    linewidth = 0) +
    
    # nice colors with viridis, add label 
    scale_fill_continuous(type = "viridis",
                          name = "symlog(Δcases)\n",
                          label = scales::comma, 
                          limits = c(-5,5)) +
    
    # highlight high risk counties
    geom_sf(data = usmap::us_map(regions = "counties", 
                          include = means$fips[means[,y] > symlog(200)]),
            aes(colour = "red"), 
            fill = NA) +
    scale_colour_manual(name = "High Risk", 
                        values = "red", 
                        labels = "Δcases > 200") + 
    
    # add title and theme 
    labs(title = title, colour = "") + 
    theme(legend.position = "right",
          plot.margin = margin(t = 5,  # Top margin
                               r = 0,  # Right margin
                               b = 0,  # Bottom margin
                               l = 0))
}

######## Plot Tab ########

# function to extract time series of cases for a given county
extract_fips <- function(fips, type_fips) {
  if(type_fips == "None") data <- covid
  else if(type_fips == "XGBoost") data <- xgboost
  else if (type_fips == "XGBoost100k") data <- xgboost100k
  else data <- covid

  return(data[data$fips == fips,])
}

# function to plot time series
time_series <- function(fips, type) {
  county <- extract_fips(fips, type)

  if(type == "None") {
    ts <- plot_ly() |>
      add_lines(x = as.Date(county$date), y = county$cases_per_100k) |>
      layout(title =
               paste("Weekly COVID cases in FIPS =", fips),
             xaxis = list(title = "Date"),
             yaxis = list(title = "Cases per 100k"))
  }

  else if(type == "XGBoost" | type == "XGBoost100k") {
    d <- as.Date(as.Date("2020-02-09") + county$days_since_zero)
    train <- county$days_since_zero < 690
    test <- county$days_since_zero >= 690

    ts <- plot_ly() |>
      add_lines(x = d[train], y = county[train,3],
                name = "y_train") |>
      add_lines(x = d[test], y = county[test,3],
                name = "y_test") |>
      add_lines(x = d, y = county[,4],
                name = "y_pred_tomorrow") |>
      add_lines(x = d, y = county[,5],
                name = "y_pred_next_week") |>
      add_lines(x = d, y = county[,6],
                name = "y_pred_next_month") |>
      add_lines(x = d, y = county[,7],
                name = "y_pred_next_3months") |>
      add_lines(x = d, y = county[,8],
                name = "y_pred_next_6months") |>
      layout(title =
               paste("Predicted Weekly COVID cases in FIPS =", fips,
                     "using", type, "model"),
             xaxis = list(title = "Date"),
             yaxis = list(title = ifelse(type == "XGBoost",
                                         "Cases", "Cases per 100k")))
  }
  
  ts
}

# run app 
shinyApp(ui = ui, server = server)
