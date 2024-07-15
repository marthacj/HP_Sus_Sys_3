# Define the folder to monitor
$folder = "C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems2\HP_Sus_Sys_2\data"

# Define the pattern to search for CSV files
$pattern = "*.csv"

# Get the current date and time
$currentDate = Get-Date

# Find files modified within the last week
$files = Get-ChildItem -Path $folder -Filter $pattern | Where-Object { $_.LastWriteTime -gt $currentDate.AddDays(-7) }

# Output the files (you can adjust this to log to a file or take other actions)
$files | ForEach-Object { Write-Output $_.FullName }
