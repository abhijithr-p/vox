<?php
session_start(); 

$conn = new mysqli("localhost", "root", "", "website_login");

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = $conn->real_escape_string($_POST['email']);
    $password = $_POST['password'];

    $sql = "SELECT * FROM users WHERE email='$email'";
    $result = $conn->query($sql);

    if ($result->num_rows == 1) {
        $user = $result->fetch_assoc();

        if (password_verify($password, $user['password'])) {
            $_SESSION['users'] = $user['username']; 

            // Redirect to home page after login
            header("Location: indexhome.html");
            exit(); // Stop further execution after redirection
        } else {
            echo "<script>alert('Incorrect password!'); window.location.href='index.signin.html';</script>";
        }
    } else {
        echo "<script>alert('User not found!'); window.location.href='index.signin.html';</script>";
    }
}

$conn->close();
?>
