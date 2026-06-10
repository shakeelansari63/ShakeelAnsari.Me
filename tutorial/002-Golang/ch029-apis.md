# REST APIs with gorilla/mux

The `gorilla/mux` package provides a powerful URL router and dispatcher. It supports path parameters, HTTP method constraints, and integrates with `net/http`.

## Setup

```go
go get -u github.com/gorilla/mux
go get -u github.com/google/uuid
```

Only one `.go` file can live in the project root (typically `main.go`). Sub-packages like `models/`, `data/`, and `controllers/` live in their own folders.

## Models

Models define the data structure. JSON struct tags control the serialized field names:

```go
// models/model.go
type Course struct {
    Id     string  `json:"courseid"`
    Name   string  `json:"coursename"`
    Price  float32 `json:"price"`
    Author *Author `json:"author"`
}

type Author struct {
    Name    string `json:"authorname"`
    Website string `json:"website"`
}

func (course *Course) IsEmpty() bool {
    return course.Name == ""
}
```

Methods on the model (`IsEmpty`) let controllers validate data without reaching into the struct directly.

## Data Layer

The data layer owns the in-memory slice and exposes CRUD functions:

```go
// data/data.go
var courses []models.Course = []models.Course{}

func InitData() {
    courses = append(courses, models.Course{
        Id:    uuid.NewString(),
        Name:  "React JS",
        Price: 249.99,
        Author: &models.Author{Name: "Hitesh", Website: "lco.dev"},
    })
}

func GetAllCourses() *[]models.Course    { return &courses }
func GetCourseById(id string) *models.Course
func AddCourse(newCourse models.Course) *models.Course
func UpdateCourse(cid string, courseToUpdate models.Course) *models.Course
func DeleteCourseById(cid string) *models.Course
```

`UpdateCourse` removes the old element by slicing around it, then appends the updated one:

```go
courses = append(courses[:idx], courses[idx+1:]...)
courses = append(courses, courseToUpdate)
```

## Controllers (HTTP Handlers)

Controllers parse the request, call the data layer, and write the response. They satisfy `http.HandlerFunc` — a function that takes `http.ResponseWriter` and `*http.Request`:

```go
// controllers/controllers.go
func GetAllCourses(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    allCourses := data.GetAllCourses()
    json.NewEncoder(w).Encode(allCourses)
}
```

For path parameters, use `mux.Vars`:

```go
func GetCourseById(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    id := vars["id"]
    course := data.GetCourseById(id)
    if course == nil {
        http.Error(w, "Error: No course found with given ID", http.StatusNotFound)
        return
    }
    json.NewEncoder(w).Encode(course)
}
```

For POST/PUT, decode the JSON body into a struct:

```go
func AddNewCourse(w http.ResponseWriter, r *http.Request) {
    if r.Body == nil {
        http.Error(w, "Error: No data provided", http.StatusBadRequest)
        return
    }
    var newCourse models.Course
    json.NewDecoder(r.Body).Decode(&newCourse)
    if newCourse.IsEmpty() {
        http.Error(w, "Error: No data provided", http.StatusBadRequest)
        return
    }
    savedCourse := data.AddCourse(newCourse)
    json.NewEncoder(w).Encode(savedCourse)
}
```

Always check `r.Body == nil` and validate the decoded struct before using it.

## Router & Server

`main.go` wires everything together:

```go
func main() {
    controllers.Initialize()         // seed dummy data
    router := mux.NewRouter()
    router.HandleFunc("/", controllers.GoToHome).Methods("GET")
    router.HandleFunc("/courses", controllers.GetAllCourses).Methods("GET")
    router.HandleFunc("/course/{id}", controllers.GetCourseById).Methods("GET")
    router.HandleFunc("/course", controllers.AddNewCourse).Methods("POST")
    router.HandleFunc("/course/{id}", controllers.UpdateCourse).Methods("PUT")
    router.HandleFunc("/course/{id}", controllers.DeleteCourse).Methods("DELETE")
    http.ListenAndServe(":4000", router)
}
```

`mux.NewRouter()` returns a router that implements `http.Handler`, so it can be passed directly to `http.ListenAndServe`. The `.Methods()` call restricts each route to a specific HTTP verb — POST to create, PUT to update, DELETE to remove.

## Summary

| Method | Endpoint          | Action       |
|--------|-------------------|--------------|
| GET    | `/`               | Home page    |
| GET    | `/courses`        | List all     |
| GET    | `/course/{id}`    | Get by ID    |
| POST   | `/course`         | Create       |
| PUT    | `/course/{id}`    | Update       |
| DELETE | `/course/{id}`    | Delete       |
